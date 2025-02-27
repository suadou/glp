from collections import namedtuple

from jax import numpy as jnp

from .periodic import make_displacement

System = namedtuple("System", ("R", "Z", "cell", "total_charge", "num_unpaired_electrons"))
UnfoldedSystem = namedtuple(
    "System", ("R", "Z", "cell", "total_charge", "num_unpaired_electrons", "mask", "replica_idx", "padding_mask", "updated")
)


def atoms_to_system(atoms, dtype=jnp.float32):
    R = jnp.array(atoms.get_positions(), dtype=dtype)
    Z = jnp.array(
        atoms.get_atomic_numbers(), dtype=jnp.int16
    )  # we will infer this type
    cell = jnp.array(atoms.get_cell().array.T, dtype=dtype)
    #TODO: delete this if/else?
    if jnp.sum(cell) == 0:
        cell = None
    else:
        cell = cell
    try:
        total_charge = atoms.info['charge']
    except:
        total_charge = jnp.array(0.)
    try:
        num_unpaired_electrons = atoms.info['multiplicity'] - 1
    except:
        num_unpaired_electrons = jnp.array(0.) 
    total_charge = jnp.array([total_charge], dtype=dtype)
    num_unpaired_electrons = jnp.array([num_unpaired_electrons], dtype=dtype)
    return System(R, Z, cell, total_charge, num_unpaired_electrons)


def unfold_system(system, unfolding):
    from glp.unfold import unfold

    N = system.R.shape[0]

    wrapped, unfolded = unfold(system.R, system.cell, unfolding)
    all_R = jnp.concatenate((wrapped, unfolded), axis=0)
    all_idx = jnp.concatenate((jnp.arange(N), unfolding.replica_idx), axis=0)
    all_Z = system.Z[all_idx]

    mask = jnp.arange(all_R.shape[0]) < N
    padding_mask = jnp.concatenate((jnp.ones(N, dtype=bool), unfolding.padding_mask))

    return UnfoldedSystem(all_R, all_Z, system.total_charge, system.num_unpaired_electrons, None, mask, all_idx, padding_mask, unfolding.updated)


def to_displacement(system):
    return make_displacement(system.cell)