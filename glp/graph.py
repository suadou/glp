from collections import namedtuple
from functools import partial

import jax
import jax.numpy as jnp

from .utils import cast
from .periodic import displacement

Graph = namedtuple("Graph", ("positions","edges", "nodes", "centers", "others", "mask", "total_charge", "num_unpaired_electrons", "edges_lr", "idx_i_lr", "idx_j_lr", "cell", "alpb" ))

def system_to_graph(system, neighbors, alpb):
    # neighbors are an *updated* neighborlist
    # question: how do we treat batching?

    positions = system.R
    nodes = system.Z

    edges = jax.vmap(partial(displacement, system.cell))(
        positions[neighbors.centers], positions[neighbors.others])
    edges_lr = jax.vmap(partial(displacement, system.cell))(
        positions[neighbors.idx_i_lr], positions[neighbors.idx_j_lr])

    mask = neighbors.centers != positions.shape[0]

    return Graph(positions, edges, nodes, neighbors.centers, neighbors.others, mask, system.total_charge, system.num_unpaired_electrons, edges_lr, neighbors.idx_i_lr, neighbors.idx_j_lr, system.cell, alpb)

