import jax.numpy as jnp
import jax.random
import pytest
from dynamiqs import basis, dag, destroy, todm

from qontrol import (
    GRAPEOptions,
    coherent_infidelity,
    control_area,
    control_norm,
    extract_info_from_h5,
    forbidden_states,
    incoherent_infidelity,
)

from .abstract_system import AbstractSystem


def _filepath(path):
    d = path / 'sub'
    d.mkdir()
    return d / 'tmp.h5py'


def setup_Kerr_osc(nH=None):
    if nH is None:
        nH = ()
    key = jax.random.PRNGKey(31)
    freq_shifts = 2.0 * jnp.pi * jax.random.normal(key, nH) / 1000
    dim = 4
    a = destroy(dim)
    H0 = -0.5 * 0.2 * dag(a) @ dag(a) @ a @ a
    H0 += jnp.einsum('...,ij->...ij', freq_shifts, dag(a) @ a)
    H1s = [a + dag(a), 1j * (a - dag(a))]
    initial_states = [basis(dim, 0), basis(dim, 1)]
    target_states = [-1j * basis(dim, 1), 1j * basis(dim, 0)]
    tsave = jnp.linspace(0, 40.0, int(40.0 // 2.0) + 1)
    return H0, H1s, tsave, initial_states, target_states


@pytest.mark.parametrize('grape_type', ['sesolve', 'mesolve'])
@pytest.mark.parametrize('infid_cost', ['coherent', 'incoherent'])
@pytest.mark.parametrize('cost', ['', 'norm', 'area', 'forbid'])
@pytest.mark.parametrize('nH', [(), (2,), (2, 3)])
def test_costs(infid_cost, grape_type, cost, nH, tmp_path):
    filepath = _filepath(tmp_path)
    H0, H1s, tsave, initial_states, target_states = setup_Kerr_osc(nH)
    options = GRAPEOptions(
        target_fidelity=0.99, epochs=4000, progress_meter=None, grape_type=grape_type
    )
    # only utilized if cost == "forbid"
    dim = H0.shape[-1]
    _forbidden_states = [basis(dim, idx) for idx in range(2, dim)]
    if grape_type == 'mesolve':
        jump_ops = [0.0001 * destroy(dim)]
        initial_states = todm(initial_states)
        target_states = todm(target_states)
        _forbidden_states = todm(_forbidden_states)
    else:
        jump_ops = []
    KerrGRAPE = AbstractSystem(
        H0, H1s, jump_ops, initial_states, target_states, tsave, options
    )
    if infid_cost == 'coherent':
        costs = [coherent_infidelity(KerrGRAPE.target_states)]
    else:
        costs = [incoherent_infidelity(KerrGRAPE.target_states)]
    if cost == '':
        pass
    elif cost == 'norm':
        costs += [control_norm(2.0 * jnp.pi * 0.005, cost_multiplier=0.1)]
    elif cost == 'area':
        costs += [control_area(cost_multiplier=0.001)]
    elif cost == 'forbid':
        forbidden_states_list = len(KerrGRAPE.initial_states) * [_forbidden_states]
        costs += [forbidden_states(forbidden_states_list=forbidden_states_list)]
    else:
        pass
    opt_params, H_t_updater = KerrGRAPE.run(costs, filepath)
    KerrGRAPE.assert_correctness(opt_params, H_t_updater, costs[0])


def test_reinitialize(tmp_path):
    filepath = _filepath(tmp_path)
    H0, H1s, tsave, initial_states, target_states = setup_Kerr_osc()
    options = GRAPEOptions(
        target_fidelity=0.99, epochs=4000, progress_meter=None, grape_type='sesolve'
    )
    KerrGRAPE = AbstractSystem(
        H0, H1s, [], initial_states, target_states, tsave, options
    )
    costs = [coherent_infidelity(KerrGRAPE.target_states)]
    _, H_t_updater = KerrGRAPE.run(costs, filepath)
    data_dict, _ = extract_info_from_h5(filepath)
    opt_params = {'dp': data_dict['dp'][-1]}  # take the last entry
    KerrGRAPE.assert_correctness(opt_params, H_t_updater, costs[0])
