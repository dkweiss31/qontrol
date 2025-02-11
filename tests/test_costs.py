from dynamiqs.random import ket as randket
from jax.random import PRNGKey

from qontrol import forbidden_states


def test_symmetric_forbidden_states():
    dim = 5
    n_states = 4
    n_forbid = 3
    forbidden_states_qarray = randket(PRNGKey(31), (n_states, n_forbid, dim, 1))
    cost = forbidden_states(forbidden_states_qarray)
    assert cost.forbidden_states.shape == (n_states, 1, n_forbid, dim, 1)


def test_ragged_forbidden_states():
    dim = 5
    n_states = 4
    forbidden_states_qarray = [
        randket(PRNGKey(31), (2, dim, 1)),
        randket(PRNGKey(32), (1, dim, 1)),
        [],
        [],
    ]
    cost = forbidden_states(forbidden_states_qarray)
    # 2 is the maximum number of forbidden states
    assert cost.forbidden_states.shape == (n_states, 1, 2, dim, 1)
