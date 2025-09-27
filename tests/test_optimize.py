import diffrax as dx
import dynamiqs as dq
import jax.numpy as jnp
import jax.random
import matplotlib
import numpy as np
import optax
import pytest
from dynamiqs.method import Expm, Tsit5
from jax import Array
from scipy.stats import unitary_group


matplotlib.use('Agg')

from qontrol import (
    coherent_infidelity,
    control_area,
    control_norm,
    extract_info_from_h5,
    forbidden_states,
    incoherent_infidelity,
    mepropagator_model,
    mesolve_model,
    optimize,
    propagator_infidelity,
    sepropagator_model,
    sesolve_model,
)
from qontrol.cost import SummedCost


def _filepath(path, suffix: str = ''):
    d = path / f'sub{suffix}'
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
        for H1, drive_param in zip(H1s, drive_params, strict=True):
            drive_spline = _drive_spline(drive_param)
            H += dq.modulated(drive_spline.evaluate, H1)
        return H

    return H_func, tsave, psi0, init_drive_params, target_states


@pytest.mark.parametrize('opt_type', ['sesolve', 'mesolve'])
@pytest.mark.parametrize('infid_cost', ['coherent', 'incoherent'])
@pytest.mark.parametrize('cost', ['', 'norm', 'area', 'forbid'])
@pytest.mark.parametrize('nH', [(), (2,), (2, 3)])
def test_costs(infid_cost, opt_type, cost, nH, tmp_path):
    filepath = _filepath(tmp_path)
    H_func, tsave, psi0, init_drive_params, target_states = setup_Kerr_osc(nH)
    optimizer_options = {'epochs': 600, 'all_costs': True, 'plot': False}
    dq_options = dq.Options(progress_meter=None)
    # only utilized if cost == "forbid"
    dim = H_func(init_drive_params).shape[-1]
    _forbidden_states = [dq.basis(dim, idx) for idx in range(2, dim)]
    if opt_type == 'mesolve':
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
        costs += 0.01 * control_norm(2.0 * jnp.pi * 0.005, target_cost=0.1)
        assert type(costs) is SummedCost
    elif cost == 'area':
        costs += 0.00001 * control_area(target_cost=0.1)
        assert type(costs) is SummedCost
    elif cost == 'forbid':
        forbidden_states_list = len(psi0) * [_forbidden_states]
        costs += 0.001 * forbidden_states(
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
    _, terminate = zip(*costs(opt_result, opt_H, opt_params), strict=True)
    assert all(terminate)


def test_reinitialize(tmp_path):
    filepath = _filepath(tmp_path)
    H_func, tsave, psi0, init_drive_params, target_states = setup_Kerr_osc()
    optimizer_options = {'epochs': 4000, 'plot': False}
    dq_options = dq.Options(progress_meter=None)
    model = sesolve_model(H_func, psi0, tsave)
    costs = coherent_infidelity(target_states, target_cost=0.01)
    optimizer = optax.adam(0.0001, b1=0.99, b2=0.99)
    optimize(
        init_drive_params,
        costs,
        model,
        filepath=filepath,
        optimizer=optimizer,
        opt_options=optimizer_options,
        dq_options=dq_options,
    )
    data_dict, _ = extract_info_from_h5(filepath)
    opt_params = {'dp': data_dict['dp'][-1]}
    opt_result, opt_H = model(opt_params, Tsit5(), None, dq_options)
    _, terminate = zip(*costs(opt_result, opt_H, opt_params), strict=True)
    assert all(terminate)


def test_save_period(tmp_path):
    filepath_1 = _filepath(tmp_path, suffix='1')
    optimizer_options_1 = {'epochs': 4000, 'plot': False, 'save_period': 1}
    data_1 = _setup_and_run(filepath_1, optimizer_options_1)
    filepath_21 = _filepath(tmp_path, suffix='21')
    optimizer_options_21 = {'epochs': 4000, 'plot': False, 'save_period': 21}
    data_21 = _setup_and_run(filepath_21, optimizer_options_21)
    for key, val_1 in data_1.items():
        assert np.allclose(val_1, data_21[key])


def _setup_and_run(filepath: str, opt_options: dict):
    H_func, tsave, psi0, init_drive_params, target_states = setup_Kerr_osc()
    dq_options = dq.Options(progress_meter=None)
    model = sesolve_model(H_func, psi0, tsave)
    costs = coherent_infidelity(target_states, target_cost=0.01)
    optimizer = optax.adam(0.0001, b1=0.99, b2=0.99)
    optimize(
        init_drive_params,
        costs,
        model,
        filepath=filepath,
        optimizer=optimizer,
        opt_options=opt_options,
        dq_options=dq_options,
    )
    data_dict, _ = extract_info_from_h5(filepath)
    return data_dict


@pytest.mark.parametrize('opt_type', ['sepropagator', 'mepropagator'])
def test_propagator(opt_type, tmp_path):
    n_times = 21
    tsave = jnp.linspace(0.0, 1.0, n_times)
    H0 = -0.5 * 2.0 * jnp.pi * dq.sigmaz()

    def H_func(parameters):
        return H0 + dq.pwc(tsave, parameters, dq.sigmax())

    rand = 31
    if opt_type == 'sepropagator':
        U_target = unitary_group.rvs(2, random_state=rand)
        model = sepropagator_model(H_func, tsave)
        dq_options = dq.Options(progress_meter=None)
    else:
        jump_ops = [1e-6 * dq.sigmam()]
        _U_target = unitary_group.rvs(2, random_state=rand)
        U_target = dq.sprepost(_U_target, _U_target.conjugate())
        model = mepropagator_model(H_func, jump_ops, tsave)
        dq_options = dq.Options()

    cost = propagator_infidelity(U_target, target_cost=1e-2)
    opt_params = optimize(
        1e-2 * jnp.ones(n_times - 1),
        cost,
        model,
        filepath=_filepath(tmp_path),
        optimizer=optax.adam(0.003, b1=0.99, b2=0.99),
        opt_options={'epochs': 1000, 'plot': False},
        dq_options=dq_options,
        method=Expm(),
    )
    opt_result, opt_H = model(opt_params, Expm(), None)
    ((cost, terminate),) = cost(opt_result, opt_H, opt_params)
    assert terminate


def setup_gate_system():
    seed_amplitude = 1e-2
    T = 500
    dt = 1
    ntimes = int(T // dt) + 1
    tsave = jnp.linspace(0, T, ntimes)

    xi01 = 0
    xi10 = 0

    sx = dq.sigmax()
    sy = dq.sigmay()
    sz = dq.sigmaz()
    eye = dq.eye(2)

    Zc1 = dq.tensor(sz, eye, eye)
    Xt = dq.tensor(eye, sx, eye)
    Zc2 = dq.tensor(eye, eye, sz)
    Yt = dq.tensor(eye, sy, eye)
    Zt = dq.tensor(eye, sz, eye)

    CX01 = dq.asqarray(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dims=(2, 2)
    )

    CX10 = dq.asqarray(
        [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dims=(2, 2)
    )

    Hk = [x * jnp.pi * 2 * 40e-3 for x in (Zc1 @ Xt, Xt @ Zc2, Xt, Yt)]
    H0 = (
        (xi01 / 4 * Zc1 @ Zt + xi10 / 4 * Zt @ Zc2 - (xi01 + xi10) / 4 * Zt)
        * jnp.pi
        * 2
    )

    def H_pwc(drive_params):
        H = H0
        for idx, _H1 in enumerate(Hk):
            H += dq.pwc(tsave, drive_params[idx], _H1)
        return H

    def t(*args):
        return dq.tensor(*args)

    target_gate = (
        t(eye, dq.hadamard(), eye)
        @ t(CX01, eye)
        @ t(eye, dq.tgate(), eye)
        @ t(eye, CX10)
        @ t(eye, dq.tgate(), eye).dag()
        @ t(CX01, eye)
        @ t(eye, dq.tgate(), eye)
        @ t(eye, CX10)
        @ t(eye, dq.tgate(), eye).dag()
        @ t(eye, dq.hadamard(), eye)
    )

    init_drive_params = seed_amplitude * jnp.ones((len(Hk), ntimes - 1))

    return H_pwc, tsave, init_drive_params, target_gate, Zt


def setup_coherent_system():
    N = 10
    seed_amplitude = 1e-2
    T = 2200
    dt = 2
    ntimes = int(T // dt) + 1
    tsave = jnp.linspace(0, T, ntimes)

    a = dq.tensor(dq.eye(2), dq.destroy(N))
    sx = dq.tensor(dq.sigmax(), dq.eye(N))
    sy = dq.tensor(dq.sigmay(), dq.eye(N))
    sm = dq.tensor(dq.sigmam(), dq.eye(N))
    dx = a + dq.dag(a)
    dy = 1j * (a - dq.dag(a))

    y0 = dq.tensor(dq.basis(2, 0), dq.basis(N, 0))
    y_target = dq.unit(
        dq.tensor(dq.basis(2, 0), dq.basis(N, 0))
        + dq.tensor(dq.basis(2, 0), dq.basis(N, 4))
    )

    pi2 = jnp.pi * 2
    kerr = -1e-6
    chi = -0.71e-3
    H0 = kerr / 2 * dq.dag(a) @ dq.dag(a) @ a @ a
    H0 += chi * dq.dag(a) @ a @ dq.dag(sm) @ sm
    H0 *= pi2

    Hk = [x * pi2 * 40e-3 for x in (sx, sy, dx, dy)]

    def H_pwc(drive_params):
        H = H0
        for idx, _H1 in enumerate(Hk):
            H += dq.pwc(tsave, drive_params[idx], _H1)
        return H

    init_drive_params = seed_amplitude * jnp.ones((len(Hk), ntimes - 1))
    exp_ops = [
        dq.tensor(dq.eye(2), dq.destroy(N))
        @ dq.dag(dq.tensor(dq.eye(2), dq.destroy(N)))
    ]

    return H_pwc, tsave, init_drive_params, [y0], [y_target], exp_ops


@pytest.mark.parametrize('learning_rate', [1e-4])
@pytest.mark.parametrize('target_cost', [1e-2])
def test_gate_plot(learning_rate, target_cost, tmp_path):
    H_pwc, tsave, init_drive_params, target_gate, _Zt = setup_gate_system()

    model = sepropagator_model(H_pwc, tsave)
    cost = propagator_infidelity(target_unitary=target_gate, target_cost=target_cost)

    optimizer = optax.adam(learning_rate=learning_rate)
    opt_options = {'verbose': False, 'plot': True, 'plot_period': 1}
    dq_options = dq.Options(progress_meter=None)

    opt_params = optimize(
        init_drive_params,
        cost,
        model,
        filepath=_filepath(tmp_path),
        optimizer=optimizer,
        opt_options=opt_options,
        method=dq.integrators.Expm(),
        dq_options=dq_options,
    )

    final_fidelity = dq.fidelity(
        dq.sepropagator(H_pwc(opt_params), tsave).propagators[-1], target_gate
    )
    assert final_fidelity >= (1 - target_cost)


@pytest.mark.parametrize('learning_rate', [1e-4])
@pytest.mark.parametrize('target_cost', [1e-2])
def test_single_target_coherent(learning_rate, target_cost, tmp_path):
    H_pwc, tsave, init_drive_params, initial_states, target_states, exp_ops = (
        setup_coherent_system()
    )

    model = sesolve_model(H_pwc, initial_states, tsave, exp_ops=exp_ops)
    cost = coherent_infidelity(target_states=target_states, target_cost=target_cost)

    optimizer = optax.adam(learning_rate=learning_rate)
    opt_options = {'verbose': False, 'plot': False}
    dq_options = dq.Options(save_states=False, progress_meter=None)

    opt_params = optimize(
        init_drive_params,
        cost,
        model,
        filepath=_filepath(tmp_path),
        optimizer=optimizer,
        opt_options=opt_options,
        method=dq.integrators.Expm(),
        dq_options=dq_options,
    )

    opt_result, opt_H = model(opt_params, dq.integrators.Expm(), None, dq_options)
    _, terminate = zip(*cost(opt_result, opt_H, opt_params), strict=True)
    assert all(terminate)
