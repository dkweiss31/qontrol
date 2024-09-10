from __future__ import annotations

import copy

from dynamiqs import unit
from jax import Array


def all_cardinal_states(basis_states: list[Array]) -> list[Array]:
    """Return a list of all cardinal states on the Bloch sphere."""
    all_states = copy.deepcopy(basis_states)
    for idx_1, state_1 in enumerate(basis_states):
        for idx_2, state_2 in enumerate(basis_states):
            if idx_2 > idx_1:
                all_states.append(unit(state_1 + state_2))
                all_states.append(unit(state_1 + 1j * state_2))
    return all_states
