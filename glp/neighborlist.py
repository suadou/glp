"""neighborlist.py

The role of a neighborlist implementation is to reduce the amount of pairs
of atoms to consider from N*N to something linear in N by removing pairs
that are farther away than a given cutoff radius.

This file implements this in two ways: 
- N-square: We first generate all N*N combinations, and then trim down these
            candidates into a fixed number of pairs. Once that number is
            decided, this procedure is jittable. The initial allocation is not.

- Cell-list: We first divide the simulation cell into a grid of cells, and 
                assign each atom to a cell. Then, for each atom, we only consider
                the atoms in the same cell and its neighbors. This is jittable. 
                The initial allocation is not.

The cell-list implementation is more efficient for large systems, but it is
more complex and requires the simulation cell to be periodic. The N-square
implementation is simpler and can be used for non-periodic systems.

The general data format consists of two index arrays, such that the indices in
`centers` contains the atom from which atom-pair vectors originate, while the
`others` contains the index which receives a given atom-pair vector.

The overall design of this is heavily inspired by jax-md by Samuel Schoenholz.

"""

from collections import namedtuple
import jax
from jax import vmap, ops
import jax.numpy as jnp
from jax.lax import stop_gradient, cond, iota
import numpy as np

from glp import comms
from .periodic import displacement
from .utils import boolean_mask_1d, cast, squared_distance

#NOTE: This was only checked in mlff simulations of cubic cells
#Assume a cubic cell to avoid subsequent computations involving inverse
#that lead to floating point errors
def get_heights(cubic_cell):
    return jnp.diag(cubic_cell)
def wrap(cubic_cell, R):
    return R % cubic_cell

#https://github.com/google/jax/issues/11321
#https://gist.github.com/tbenthompson/faae311ec4e465b0ff47b4aabe0d56b2
def inv33(mat):
    m1, m2, m3 = mat[0]
    m4, m5, m6 = mat[1]
    m7, m8, m9 = mat[2]
    det = m1 * (m5 * m9 - m6 * m8) + m4 * (m8 * m3 - m2 * m9) + m7 * (m2 * m6 - m3 * m5)
    inv_det = 1.0 / det
    return (
        jnp.array(
            [
                [m5 * m9 - m6 * m8, m3 * m8 - m2 * m9, m2 * m6 - m3 * m5],
                [m6 * m7 - m4 * m9, m1 * m9 - m3 * m7, m3 * m4 - m1 * m6],
                [m4 * m8 - m5 * m7, m2 * m7 - m1 * m8, m1 * m5 - m2 * m4],
            ]
        )
        * inv_det
    )

Neighbors = namedtuple(
    "Neighbors", ("centers", "others", "idx_i_lr", "idx_j_lr", "overflow", "reference_positions", "cell_list")
)

CellList = namedtuple(
    "CellList", ("id", "reallocate", "capacity", "size", "cells_per_side")
)

def neighbor_list(system, cutoff, skin, capacity_multiplier=1.25, buffer_size_multiplier=1.25):
    # convenience interface: we often don't need explicit access to an allocate_fn, since
    #     the neighborlist setup takes that role (see for instance all the calculators)
    #     so this just does the allocation and gives us a jittable update function back

    allocate, update = quadratic_neighbor_list(
        system.cell, cutoff, skin, capacity_multiplier=capacity_multiplier, buffer_size_multiplier=buffer_size_multiplier
    )

    if hasattr(system, "padding_mask"):

        def _update(system, neighbors):
            neighbors = update(
                system.R,
                neighbors,
                new_cell=system.cell,
                padding_mask=system.padding_mask,
                force_update=system.updated,
            )
            return neighbors

        neighbors = allocate(system.R, padding_mask=system.padding_mask)
    else:

        def _update(system, neighbors):
            neighbors = update(system.R, neighbors, new_cell=system.cell)
            return neighbors

        neighbors = allocate(system.R)

    return neighbors, _update


def quadratic_neighbor_list(cell, cutoff,  skin, capacity_multiplier=1.25, use_cell_list=False, debug=False, lr_cutoff=0, buffer_size_multiplier=1.25):
    """Implementation of neighborlist in pbc using cell list."""

    assert capacity_multiplier >= 1.0

    cell = stop_gradient(cell)

    short_cutoff = cast(stop_gradient(cutoff))
    lr_cutoff = cast(stop_gradient(lr_cutoff))
    if lr_cutoff > 0:
        short_cutoff = jnp.minimum(short_cutoff, lr_cutoff)
        cutoff = jnp.maximum(short_cutoff, lr_cutoff)
    else:
        cutoff = short_cutoff
    skin = cast(stop_gradient(skin))
    if cell is not None:

        def cell_too_small(new_cell):
            min_height = jnp.min(get_heights(new_cell))
            return cast(2) * (cutoff + skin) > min_height

    else:

        def cell_too_small(new_cell):
            return False

    if cell_too_small(cell):
        min_height = jnp.min(get_heights(cell))
        comms.warn(
            f"warning: total cutoff {(cutoff+skin):.2f} does not fit within simulation cell (needs to be < {0.5*min_height:.2f})"
        )
        comms.warn("this will yield incorrect results!")

    cutoff = (cutoff + skin)

    allowed_movement = (skin * cast(0.5)) ** cast(2.0)

    if cell is not None and use_cell_list:
        if jnp.all(jnp.any(cutoff < get_heights(cell) / 3., axis=0)):
            cl_allocate, cl_update = cell_list(cell, cutoff, buffer_size_multiplier)
        else:
            cl_allocate = cl_update = None
    else:
        cl_allocate = cl_update = None
        
    def need_update_fn(neighbors, new_positions, new_cell):
        # question: how to deal with changes in new_cell?
        # we will invalidate if atoms move too much, but not if
        # the new cell is too small for the cutoff...

        movement = make_squared_distance(new_cell)(
            neighbors.reference_positions, new_positions
        )
        max_movement = jnp.max(movement)

        # if we have an overflow, we don't need to update -- results are
        # already invalid. instead, we will simply return previous invalid
        # neighbors and hope someone up the chain catches it!
        should_update = (max_movement > allowed_movement) & (~neighbors.overflow) 

        # if the cell is now *too small*, we need to update for sure to set overflow flag
        return should_update | cell_too_small(new_cell)

    def allocate_fn(positions, new_cell=None, padding_mask=None):
        # this is not jittable,
        # we're determining shapes

        if new_cell is None:
            new_cell = cell
        else:
            new_cell = stop_gradient(new_cell)

        N = positions.shape[0]
        cl = cl_allocate(positions)  if cl_allocate is not None else None
        if cl is not None:
            comms.warn(f"suggestion: N is {(N):.0f}, cl.id is {(cl.id.size):.0f} consider using different buffer_size_multiplier, so that cl.id is slightly bigger than N. Now it is {(buffer_size_multiplier):.0f})")
        centers, others, mask, mask_pairs, hits, hits_pairs = get_neighbors_allocate(
            positions,
            make_squared_distance(new_cell),
            short_cutoff,
            lr_cutoff,
            padding_mask=padding_mask,
            cl=cl
        )

        size_lr = int(hits_pairs.item() * capacity_multiplier + 1)
        idx_i_lr, _ = boolean_mask_1d(centers, mask_pairs, size_lr, N)
        idx_j_lr, _ = boolean_mask_1d(others, mask_pairs, size_lr, N)

        size_sr = int(hits.item() * capacity_multiplier + 1)
        centers, _ = boolean_mask_1d(centers, mask, size_sr, N)
        others, overflow = boolean_mask_1d(others, mask, size_sr, N)
        overflow = overflow | cell_too_small(new_cell)

        return Neighbors(centers, others, idx_i_lr, idx_j_lr, overflow, positions, cl)
    
    def update_fn(positions, neighbors, new_cell=None, padding_mask=None, force_update=False):
        # this is jittable,
        # neighbors tells us all the shapes we need
        if new_cell is None:
            new_cell = cell
        else:
            new_cell = stop_gradient(new_cell)
        
        
        N = positions.shape[0]
        size = neighbors.centers.shape[0]
        size_pairs = neighbors.idx_i_lr.shape[0]
        def update(positions, cell, padding_mask, cl=neighbors.cell_list):
            cl = cl_update(positions, cl, new_cell)  if cl_update is not None else None
            centers, others, mask, mask_pairs = get_neighbors_update(
                positions,
                make_squared_distance(cell),
                short_cutoff,
                lr_cutoff,
                padding_mask=padding_mask,
                cl=cl
            )
            idx_i_lr, _ = boolean_mask_1d(centers, mask_pairs, size_pairs, N)
            idx_j_lr, _ = boolean_mask_1d(others, mask_pairs, size_pairs, N)
            centers, _ = boolean_mask_1d(centers, mask, size, N)
            others, overflow = boolean_mask_1d(others, mask, size, N)
            overflow = overflow | cell_too_small(cell)
            return Neighbors(centers, others, idx_i_lr, idx_j_lr, overflow, positions, cl)

        # if we need an update, call update(), else do a no-op and return input
        return cond(
            force_update | need_update_fn(neighbors, positions, new_cell),
            update,
            lambda a, b, c: neighbors,
            positions,
            new_cell,
            padding_mask,
        )

    if debug:
        return allocate_fn, update_fn, need_update_fn
    else:
        return allocate_fn, update_fn


def candidates_fn(n):
    candidates = jnp.arange(n)
    square = jnp.broadcast_to(candidates[None, :], (n, n))
    centers = jnp.reshape(jnp.transpose(square), (-1,))
    others = jnp.reshape(square, (-1,))
    return centers, others

def cell_list_candidate_fn(cell_id, N, dim):
    idx = cell_id
    cell_idx = [idx] * (dim**3)

    for i, dindex in enumerate(neighboring_cells(dim)):
        cell_idx[i] = shift_array(idx, dindex)

    cell_idx = jnp.concatenate(cell_idx, axis=-2)
    cell_idx = cell_idx[..., jnp.newaxis, :, :]
    cell_idx = jnp.broadcast_to(cell_idx, idx.shape[:-1] + cell_idx.shape[-2:])

    def copy_values_from_cell(value, cell_value, cell_id):
        scatter_indices = jnp.reshape(cell_id, (-1,))
        cell_value = jnp.reshape(cell_value, (-1,) + cell_value.shape[-2:])
        return value.at[scatter_indices].set(cell_value)

    neighbor_idx = jnp.zeros((N + 1,) + cell_idx.shape[-2:], jnp.int32)
    neighbor_idx = copy_values_from_cell(neighbor_idx, cell_idx, idx)
    others = jnp.reshape(neighbor_idx[:-1, :, 0], (-1,))
    centers = jnp.repeat(jnp.arange(0, N), others.shape[0] // N)
    return centers, others

# def get_neighbors(positions, square_distances, cutoff, lr_cutoff, padding_mask=None, cl=None):
#     N, dim = positions.shape
#     if cl is not None:
#         centers, others = cell_list_candidate_fn(cl.id, N, dim)
#     else:
#         centers, others = candidates_fn(N)

#     sq_distances = square_distances(positions[centers], positions[others])
#     mask = sq_distances <= (cutoff**2)
#     mask = mask * ((centers != others) & (others<N))  # remove self-interactions and neighbors repetitions

#     if padding_mask is not None:
#         mask = mask * padding_mask[centers]
#         mask = mask * padding_mask[others]
    
#     mask_pairs = sq_distances <= (lr_cutoff**2)
#     mask_pairs = mask_pairs * ((centers != others) & (others<N))  # remove self-interactions and neighbors repetitions

#     hits = jnp.sum(mask)
#     hits_pairs = jnp.sum(mask_pairs)

#     return centers, others, sq_distances, mask, mask_pairs, hits, hits_pairs

#Separate get_neighbors into allocate and update to avoid some redundant computations/returns
def get_neighbors_allocate(positions, square_distances, cutoff, lr_cutoff, padding_mask=None, cl=None):
    N, dim = positions.shape
    if cl is not None:
        centers, others = cell_list_candidate_fn(cl.id, N, dim)
    else:
        centers, others = candidates_fn(N)

    sq_distances = square_distances(positions[centers], positions[others])
    mask = sq_distances <= (cutoff**2)
    mask = mask * ((centers != others) & (others<N))  # remove self-interactions and neighbors repetitions

    if padding_mask is not None:
        mask = mask * padding_mask[centers]
        mask = mask * padding_mask[others]
    
    mask_pairs = sq_distances <= (lr_cutoff**2)
    mask_pairs = mask_pairs * ((centers != others) & (others<N))  # remove self-interactions and neighbors repetitions

    hits = jnp.sum(mask)
    hits_pairs = jnp.sum(mask_pairs)

    return centers, others, mask, mask_pairs, hits, hits_pairs

def get_neighbors_update(positions, square_distances, cutoff, lr_cutoff, padding_mask=None, cl=None):
    N, dim = positions.shape
    if cl is not None:
        centers, others = cell_list_candidate_fn(cl.id, N, dim)
    else:
        centers, others = candidates_fn(N)

    sq_distances = square_distances(positions[centers], positions[others])
    mask = sq_distances <= (cutoff**2)
    mask = mask * ((centers != others) & (others<N))  # remove self-interactions and neighbors repetitions

    if padding_mask is not None:
        mask = mask * padding_mask[centers]
        mask = mask * padding_mask[others]
    
    mask_pairs = sq_distances <= (lr_cutoff**2)
    mask_pairs = mask_pairs * ((centers != others) & (others<N))  # remove self-interactions and neighbors repetitions

    return centers, others, mask, mask_pairs

def make_squared_distance(cell):
    return vmap(lambda Ra, Rb: squared_distance(displacement(cell, Ra, Rb)))


def cell_list(cell, cutoff, buffer_size_multiplier=1.25, bin_size_multiplier=1):

    cell = jnp.array(cell)
    cutoff *= bin_size_multiplier
    def allocate_fn(positions, extra_capacity=0):
        # This function is not jittable, we're determining shapes
        N = positions.shape[0]
        _, cell_size, cells_per_side, cell_count = cell_dimensions(cell, cutoff)
        cell_capacity = estimate_cell_capacity(positions, cell, cell_size, buffer_size_multiplier)
        cell_capacity += extra_capacity
        overflow = False
        cell_id = N * jnp.ones((cell_count * cell_capacity, 1), dtype=jnp.int32)
        particle_id = iota(jnp.int32, N)
        indices = jnp.array(jnp.floor(jnp.dot(positions, inv33(cell_size).T)), dtype=jnp.int32)

        # Some particles are in the edge and might have negative indices or larger than cells_per_side
        # We need to correct them wrapping into cell per side vector
        indices = jnp.rint(wrap(cells_per_side, indices))#.astype(jnp.int32)

        hash_multipliers = compute_hash_constants(cells_per_side)
        hashes = jnp.sum(indices * hash_multipliers, axis=1, dtype=jnp.int32)

        sort_map = jnp.argsort(hashes)
        sorted_hash = hashes[sort_map]
        sorted_id = particle_id[sort_map]

        sorted_cell_id = jnp.mod(iota(jnp.int32, N), cell_capacity)
        sorted_cell_id = sorted_hash * cell_capacity + sorted_cell_id
        sorted_id = jnp.reshape(sorted_id, (N, 1))

        cell_id = cell_id.at[sorted_cell_id].set(sorted_id)
        cell_id = unflatten_cell_buffer(cell_id, cells_per_side)

        occupancy = ops.segment_sum(jnp.ones_like(hashes), hashes, cell_count)
        max_occupancy = jnp.max(occupancy)
        overflow = overflow | (max_occupancy > cell_capacity)

        return CellList(cell_id, overflow, cell_capacity, cell_size, cells_per_side)
    
    def update_fn(positions, old_cell_list, new_cell):
        # this is jittable,
        # CellList tells us all the shapes we need
        # If cell size is lower than cutoff, we need to reallocate cell list
        # Rellocate is not jitable, we're changing shapes
        # So, after each update we need to check if we need to reallocate
        N = positions.shape[0]

        cell_size = jnp.where(old_cell_list.cells_per_side != 0, cell / old_cell_list.cells_per_side, cell)
        #TODO: check if we need to check max_occupancy every update, seems excessive
        max_occupancy = estimate_cell_capacity(positions, new_cell, cell_size, buffer_size_multiplier)
        # Checking if update or reallocate
        reallocate = jnp.all(get_heights(cell_size).any() >= cutoff) & (max_occupancy <= old_cell_list.capacity)

        def update(positions, old_cell_list):
            hash_multipliers = compute_hash_constants(old_cell_list.cells_per_side)
            indices = jnp.array(jnp.floor(jnp.dot(positions, inv33(cell_size).T)), dtype = jnp.int32)

            # Some particles are in the edge and might have negative indices or larger than cells_per_side
            # We need to correct them wrapping into cell per side vector
            indices = jnp.rint(wrap(old_cell_list.cells_per_side, indices))#.astype(jnp.int32)
            
            hashes = jnp.sum(indices * hash_multipliers, axis=1, dtype=jnp.int32)
            particle_id = iota(jnp.int32, N)
            sort_map = jnp.argsort(hashes)
            sorted_hash = hashes[sort_map]
            sorted_id = particle_id[sort_map]

            sorted_cell_id = jnp.mod(iota(jnp.int32, N), old_cell_list.capacity)
            sorted_cell_id = sorted_hash * old_cell_list.capacity + sorted_cell_id
            sorted_id = jnp.reshape(sorted_id, (N, 1))
            cell_id = N * jnp.ones((old_cell_list.id.reshape(-1, 1).shape), dtype = jnp.int32)
            cell_id = cell_id.at[sorted_cell_id].set(sorted_id)

            # This is not jitable, we're changing shapes. It's a fix for the unflatten_cell_buffer.
            
            cell_id = jax.pure_callback(unflatten_cell_buffer, old_cell_list.id, cell_id, old_cell_list.cells_per_side)
            return CellList(cell_id, old_cell_list.reallocate, old_cell_list.capacity, cell_size, old_cell_list.cells_per_side)
        
        # In case cell size is lower than cutoff, we need to reallocate 
        def need_reallocate(_ ,old_cell_list):
            return CellList(old_cell_list.id, True, old_cell_list.capacity, cell_size, old_cell_list.cells_per_side)
        return cond(reallocate, need_reallocate, update, positions, old_cell_list)
    return allocate_fn, update_fn 


def estimate_cell_capacity(positions, cell, cell_size, buffer_size_multiplier):
    minimum_cell_size = jnp.min(get_heights(cell_size))
    cell_capacity = jnp.max(count_cell_filling(positions, cell, minimum_cell_size))
    return (cell_capacity * buffer_size_multiplier).astype(jnp.int32)

def cell_dimensions(cell, cutoff):
    """Compute the number of cells-per-side and total number of cells in a box."""
    # Considering cell is 3x3 array
    # Transform into reciprocal space to get the cell size whatever cell is
    face_dist = get_heights(cell).astype(jnp.int32)
    cells_per_side = jnp.array(jnp.floor(face_dist / cutoff), dtype = jnp.int32)
    cell_size = jnp.where(cells_per_side != 0, cell / cells_per_side, cell)
    cell_count = jnp.prod(cells_per_side).astype(jnp.int32)
    return cell, cell_size, cells_per_side, cell_count

    
def count_cell_filling(position, cell, minimum_cell_size):
    """
    Counts the number of particles per-cell in a spatial partitioning scheme.
    """
    cell, cell_size, cells_per_side, _ = cell_dimensions(cell, minimum_cell_size)
    hash_multipliers = compute_hash_constants(cells_per_side)
    particle_index = jnp.array(jnp.floor(jnp.dot(position, inv33(cell_size).T)), dtype=jnp.int32)
    particle_hash = jnp.sum(particle_index * hash_multipliers, axis=1)
    filling = jnp.zeros_like(particle_hash, dtype=jnp.int32) # jnp.zeros((cell_count,), dtype=jnp.int32)
    filling = filling.at[particle_hash].add(1)
    return filling

def compute_hash_constants(cells_per_side):
    one = jnp.array([1])
    cells_per_side = jnp.concatenate((one, cells_per_side[:-1]), axis=0)
    return jnp.array(jnp.cumprod(cells_per_side), dtype=jnp.int32)

def unflatten_cell_buffer(arr, cells_per_side):
    cells_per_side = tuple(cells_per_side)
    return jnp.reshape(arr, cells_per_side + (-1,) + arr.shape[1:])

def neighboring_cells(dimension):
    for dindex in np.ndindex(*([3] * dimension)):
        yield jnp.array(dindex) - 1

def shift_array(arr, dindex):
    dx, dy, dz = tuple(dindex) + (0,) * (3 - len(dindex))
    arr = cond(dx < 0,
               lambda x: jnp.concatenate((x[1:], x[:1])),
               lambda x: cond(dx > 0,
                              lambda x: jnp.concatenate((x[-1:], x[:-1])),
                              lambda x: x, arr),
               arr)

    arr = cond(dy < 0,
               lambda x: jnp.concatenate((x[:, 1:], x[:, :1]), axis=1),
               lambda x: cond(dy > 0,
                              lambda x: jnp.concatenate((x[:, -1:], x[:, :-1]), axis=1),
                              lambda x: x, arr),
               arr)

    arr = cond(dz < 0,
               lambda x: jnp.concatenate((x[:, :, 1:], x[:, :, :1]), axis=2),
               lambda x: cond(dz > 0,
                              lambda x: jnp.concatenate((x[:, :, -1:], x[:, :, :-1]), axis=2),
                              lambda x: x, arr),
               arr)

    return arr


# I had to include it for reallocate cell list, calling cell_list allocate not jitable
def check_reallocation_cell_list(cell_list, allocate_fn, positions):
    if cell_list.reallocate:
        return allocate_fn(positions)
    return cell_list