import dynamiqs as dq
from jax.random import PRNGKey

import qontrol as ql


def test_symmetric_forbidden_states():
    dim = 5
    n_states = 4
    n_forbid = 3
    forbidden_states_qarray = dq.random.ket(PRNGKey(31), (n_states, n_forbid, dim, 1))
    cost = ql.forbidden_states(forbidden_states_qarray)
    assert cost.forbidden_states.shape == (n_states, 1, n_forbid, dim, 1)


def test_ragged_forbidden_states():
    dim = 5
    n_states = 4
    forbidden_states_qarray = [
        dq.random.ket(PRNGKey(31), (2, dim, 1)),
        dq.random.ket(PRNGKey(32), (1, dim, 1)),
        [],
        [],
    ]
    cost = ql.forbidden_states(forbidden_states_qarray)
    # 2 is the maximum number of forbidden states
    assert cost.forbidden_states.shape == (n_states, 1, 2, dim, 1)


def test_mul():
    costs = [
        ql.incoherent_infidelity([dq.basis(2, 0)]),
        ql.custom_cost(lambda _r, _H, _p: 0.3, 1.0, 0.0),
    ]
    for cost in costs:
        new_cost = 2.0 * cost
        assert new_cost.cost_multiplier == 2.0
