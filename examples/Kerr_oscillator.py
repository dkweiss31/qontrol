import argparse
from functools import partial

import diffrax as dx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from dynamiqs import basis, dag, destroy, sesolve, timecallable, modulated
from jax import Array
import jax.tree_util as jtu

from optamiqs import GRAPEOptions, all_cardinal_states, generate_file_path, grape, PulseOptimizer
from optamiqs import IncoherentInfidelity, ForbiddenStates

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GRAPE sim')
    parser.add_argument('--dim', default=5, type=int, help='tmon hilbert dim cutoff')
    parser.add_argument(
        '--Kerr', default=0.100, type=float, help='transmon Kerr in GHz'
    )
    parser.add_argument('--dt', default=2.0, type=float, help='time step for controls')
    parser.add_argument('--time', default=14.0, type=float, help='gate time')
    parser.add_argument(
        '--ramp_nts', default=3, type=int, help='numper of points in ramps'
    )
    parser_args = parser.parse_args()
    filename = generate_file_path('h5py', 'Kerr_tc', 'out')

    optimizer = optax.adam(learning_rate=0.0001, b1=0.999, b2=0.999)
    ntimes = int(parser_args.time // parser_args.dt) + 1
    tsave = jnp.linspace(0, parser_args.time, ntimes)

    # force the control endpoints to be at zero
    cos_ramp = (1 - jnp.cos(jnp.linspace(0.0, jnp.pi, parser_args.ramp_nts))) / 2
    envelope = jnp.concatenate(
        (cos_ramp, jnp.ones(ntimes - 2 * parser_args.ramp_nts), jnp.flip(cos_ramp))
    )
    options = GRAPEOptions(
        save_states=True,
        progress_meter=None,
        target_fidelity=0.995,
        epochs=10000,
    )
    a = destroy(parser_args.dim)
    H0 = -0.5 * parser_args.Kerr * 2.0 * jnp.pi * dag(a) @ dag(a) @ a @ a
    H1 = [a + dag(a), 1j * (a - dag(a))]
    H1_labels = ['I', 'Q']

    initial_states = [basis(parser_args.dim, 0), basis(parser_args.dim, 1)]
    final_states = [basis(parser_args.dim, 1), basis(parser_args.dim, 0)]
    _forbidden_states = [basis(parser_args.dim, idx)
                         for idx in range(2, parser_args.dim)]

    # need to form superpositions so that the phase information is correct
    if not options.coherent:
        initial_states = all_cardinal_states(initial_states)
        final_states = all_cardinal_states(final_states)

    forbidden_states = len(initial_states) * [_forbidden_states,]
    init_drive_params = 0.001 * jnp.ones((len(H1), ntimes))

    def _drive_spline(
        drive_params: Array, envelope: Array, ts: Array
    ) -> dx.CubicInterpolation:
        # note swap of axes so that time axis is first
        drive_w_envelope = jnp.einsum('t,dt->td', envelope, drive_params)
        drive_coeffs = dx.backward_hermite_coefficients(ts, drive_w_envelope)
        return dx.CubicInterpolation(ts, drive_coeffs)

    def H_func(t: float, drive_params: Array, envelope: Array, ts: Array) -> Array:
        drive_spline = _drive_spline(drive_params, envelope, ts)
        drive_amps = drive_spline.evaluate(t)
        drive_Hs = jnp.einsum('d,dij->ij', drive_amps, H1)
        return H0 + drive_Hs

    H_tc = jtu.Partial(H_func, envelope=envelope, ts=tsave)

    def update_fun(H, drive_params):
        H_func = jtu.Partial(H, drive_params=drive_params)
        return timecallable(H_func)

    pulse_optimizer = PulseOptimizer(H_tc, update_fun)

    costs = [
        IncoherentInfidelity(target_states=final_states, cost_multiplier=1.0),
        ForbiddenStates(forbidden_states=forbidden_states, cost_multiplier=5.0),
    ]

    opt_params = grape(
        pulse_optimizer,
        initial_states=initial_states,
        tsave=tsave,
        costs=costs,
        params_to_optimize=init_drive_params,
        filepath=filename,
        optimizer=optimizer,
        options=options,
        init_params_to_save=parser_args.__dict__,
    )

    finer_times = jnp.linspace(0.0, parser_args.time, 201)
    drive_spline = _drive_spline(opt_params, envelope, tsave)
    drive_amps = jnp.asarray(
        [drive_spline.evaluate(t) for t in finer_times]
    ).swapaxes(0, 1)
    fig, ax = plt.subplots()
    for drive_idx in range(len(H1)):
        plt.plot(
            finer_times,
            drive_amps[drive_idx] / (2.0 * np.pi),
            label=H1_labels[drive_idx],
        )
    ax.set_xlabel('time [ns]')
    ax.set_ylabel('pulse amplitude [GHz]')
    ax.legend()
    plt.tight_layout()
    plt.show()

    H_func = partial(H_func, drive_params=opt_params, envelope=envelope, ts=tsave)
    H_tc = timecallable(H_func)
    plot_result = sesolve(
        H_tc,
        initial_states,
        finer_times,
        exp_ops=[basis(parser_args.dim, idx)
                 @ dag(basis(parser_args.dim, idx))
                 for idx in range(parser_args.dim)],
        options=options,
    )
    init_labels = [
        r'$|0\rangle$',
        r'$|0\rangle+|1\rangle$',
        r'$|0\rangle+i|1\rangle$',
        r'$|1\rangle$',
    ]
    exp_labels = [r'$|0\rangle$', r'$|1\rangle$', r'$|2\rangle$', r'$|3\rangle$']

    # for brevity only plot one initial state
    state_idx_to_plot = 0
    fig, ax = plt.subplots()
    expects = plot_result.expects[state_idx_to_plot]
    for e_result, label in zip(expects, exp_labels):
        plt.plot(finer_times, e_result, label=label)
    ax.legend()
    ax.set_xlabel('time [ns]')
    ax.set_ylabel('population')
    ax.set_title(f'initial state={init_labels[state_idx_to_plot]}')
    plt.show()
