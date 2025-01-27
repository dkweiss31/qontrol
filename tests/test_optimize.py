import diffrax as dx
import dynamiqs as dq
import jax.numpy as jnp
import jax.random
import optax
import pytest
from dynamiqs.solver import Tsit5
from jax import Array

from qontrol import (
    coherent_infidelity,
    control_area,
    control_norm,
    extract_info_from_h5,
    forbidden_states,
    incoherent_infidelity,
    mesolve_model,
    optimize,
    sesolve_model,
)
from qontrol.cost import SummedCost


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
    a = dq.destroy(dim)
    H0 = -0.5 * 0.2 * dq.dag(a) @ dq.dag(a) @ a @ a
    H0 += freq_shifts[..., None, None] * dq.dag(a) @ a
    H1s = [a + dq.dag(a), 1j * (a - dq.dag(a))]
    psi0 = [dq.basis(dim, 0), dq.basis(dim, 1)]
    target_states = [-1j * dq.basis(dim, 1), 1j * dq.basis(dim, 0)]
    tsave = jnp.linspace(0, 40.0, int(40.0 // 2.0) + 1)

    init_drive_params = {'dp': -0.001 * jnp.ones((len(H1s), len(tsave)))}

    def _drive_spline(drive_params: Array) -> dx.CubicInterpolation:
        drive_coeffs = dx.backward_hermite_coefficients(tsave, drive_params)
        return dx.CubicInterpolation(tsave, drive_coeffs)

    def H_func(drive_params_dict: dict) -> Array:
        drive_params = drive_params_dict['dp']
        H = H0
        for H1, drive_param in zip(H1s, drive_params):
            drive_spline = _drive_spline(drive_param)
            H += dq.modulated(drive_spline.evaluate, H1)
        return H

    return H_func, tsave, psi0, init_drive_params, target_states


@pytest.mark.parametrize('grape_type', ['sesolve', 'mesolve'])
@pytest.mark.parametrize('infid_cost', ['coherent', 'incoherent'])
@pytest.mark.parametrize('cost', ['', 'norm', 'area', 'forbid'])
@pytest.mark.parametrize('nH', [(), (2,), (2, 3)])
def test_costs(infid_cost, grape_type, cost, nH, tmp_path):
    filepath = _filepath(tmp_path)
    H_func, tsave, psi0, init_drive_params, target_states = setup_Kerr_osc(nH)
    optimizer_options = {'epochs': 600, 'all_costs': True, 'plot': False}
    dq_options = dq.Options(progress_meter=None)
    # only utilized if cost == "forbid"
    dim = H_func(init_drive_params).shape[-1]
    _forbidden_states = [dq.basis(dim, idx) for idx in range(2, dim)]
    if grape_type == 'mesolve':
        jump_ops = [0.0001 * dq.destroy(dim)]
        psi0 = dq.todm(psi0)
        target_states = dq.todm(target_states)
        _forbidden_states = dq.todm(_forbidden_states)
        model = mesolve_model(H_func, jump_ops, psi0, tsave)
    else:
        model = sesolve_model(H_func, psi0, tsave)
    if infid_cost == 'coherent':
        costs = coherent_infidelity(target_states, target_cost=0.01)
    else:
        costs = incoherent_infidelity(target_states, target_cost=0.01)
    if cost == '':
        pass
    elif cost == 'norm':
        costs += 0.1 * control_norm(2.0 * jnp.pi * 0.005, target_cost=0.1)
        assert type(costs) is SummedCost
    elif cost == 'area':
        costs += 0.00001 * control_area(target_cost=0.1)
        assert type(costs) is SummedCost
    elif cost == 'forbid':
        forbidden_states_list = len(psi0) * [_forbidden_states]
        costs += 0.01 * forbidden_states(
            forbidden_states_list=forbidden_states_list, target_cost=0.1
        )
        assert type(costs) is SummedCost
    else:
        pass
    costs *= 1.0  # test multiplying Costs or SummedCosts
    optimizer = optax.adam(0.001, b1=0.99, b2=0.99)
    opt_params = optimize(
        init_drive_params,
        costs,
        model,
        filepath=filepath,
        optimizer=optimizer,
        opt_options=optimizer_options,
        dq_options=dq_options,
    )
    opt_result, opt_H = model(opt_params, Tsit5(), None, dq_options)
    cost_values, terminate = zip(*costs(opt_result, opt_H, opt_params))
    assert all(terminate)


def test_reinitialize(tmp_path):
    filepath = _filepath(tmp_path)
    H_func, tsave, psi0, init_drive_params, target_states = setup_Kerr_osc()
    optimizer_options = {'epochs': 4000, 'plot': False}
    dq_options = dq.Options(progress_meter=None)
    model = sesolve_model(H_func, psi0, tsave)
    costs = coherent_infidelity(target_states, target_cost=0.01)
    optimizer = optax.adam(0.0001, b1=0.99, b2=0.99)
    opt_params = optimize(
        init_drive_params,
        costs,
        model,
        filepath=filepath,
        optimizer=optimizer,
        opt_options=optimizer_options,
        dq_options=dq_options,
    )
    data_dict, _ = extract_info_from_h5(filepath)
    opt_result, opt_H = model(opt_params, Tsit5(), None, dq_options)
    cost_values, terminate = zip(*costs(opt_result, opt_H, opt_params))
    assert all(terminate)
