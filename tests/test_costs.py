from dynamiqs.random import ket as randket
from jax.random import PRNGKey

from qontrol import forbidden_states


def test_symmetric_forbidden_states():
    dim = 5
    n_states = 4
    shape = (n_states, n_states - 1, dim, 1)
    forbidden_states_array = randket(PRNGKey(31), shape)
    cost = forbidden_states(forbidden_states_array)
    assert cost.forbidden_states.shape == shape


def test_ragged_forbidden_states():
    dim = 5
    n_states = 3
    forbidden_states_array = [
        randket(PRNGKey(31), (2, dim, 1)),
        randket(PRNGKey(32), (1, dim, 1)),
        [],
    ]
    cost = forbidden_states(forbidden_states_array)
    assert cost.forbidden_states.shape == (n_states, 2, dim, 1)
