import argparse

import diffrax as dx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from dynamiqs import basis, dag, sesolve, modulated, sigmax, sigmay, sigmaz
from jax import Array

from optamiqs import GRAPEOptions, all_cardinal_states, generate_file_path, grape, PulseOptimizer
from optamiqs import IncoherentInfidelity, ControlNorm, ControlArea

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GRAPE sim')
    parser.add_argument('--dt', default=2.0, type=float, help='time step for controls')
    parser.add_argument('--time', default=30.0, type=float, help='gate time')
    parser.add_argument(
        '--ramp_nts', default=3, type=int, help='numper of points in ramps'
    )
    parser_args = parser.parse_args()
    filename = generate_file_path('h5py', 'qubit_modulated', 'out')

    optimizer = optax.adam(learning_rate=0.0001, b1=0.99, b2=0.99)
    ntimes = int(parser_args.time // parser_args.dt) + 1
    tsave = jnp.linspace(0, parser_args.time, ntimes)

    # force the control endpoints to be at zero
    cos_ramp = (1 - jnp.cos(jnp.linspace(0.0, jnp.pi, parser_args.ramp_nts))) / 2
    envelope = jnp.concatenate(
        (cos_ramp, jnp.ones(ntimes - 2 * parser_args.ramp_nts), jnp.flip(cos_ramp))
    )
    H0 = 0.0 * sigmaz()
    H1s = [sigmax(), sigmay()]
    H1_labels = ['X', 'Y']

    initial_states = [basis(2, 0), basis(2, 1)]
    final_states = [1j * basis(2, 1), -1j * basis(2, 0)]

    # need to form superpositions so that the phase information is correct
    initial_states = all_cardinal_states(initial_states)
    final_states = all_cardinal_states(final_states)

    init_drive_params = {"dp": 0.001 * jnp.ones((len(H1s), ntimes))}

    def _drive_spline(
        drive_params: Array, envelope: Array, ts: Array
    ) -> dx.CubicInterpolation:
        # note swap of axes so that time axis is first
        drive_w_envelope = jnp.einsum('t,...t->t...', envelope, drive_params)
        drive_coeffs = dx.backward_hermite_coefficients(ts, drive_w_envelope)
        return dx.CubicInterpolation(ts, drive_coeffs)

    def H_func(drive_params_dict: Array) -> Array:
        drive_params = drive_params_dict["dp"]
        H = H0
        for H1, drive_param in zip(H1s, drive_params):
            drive_spline = _drive_spline(drive_param, envelope, tsave)
            H += modulated(drive_spline.evaluate, H1)
        return H

    pulse_optimizer = PulseOptimizer(H_func, lambda H, dp: H(dp))

    costs = [IncoherentInfidelity(target_states=final_states, cost_multiplier=1.0),
             ControlNorm(cost_multiplier=1.0),
             # ControlArea(cost_multiplier=1.0),
             ]

    opt_params = grape(
        pulse_optimizer,
        initial_states=initial_states,
        tsave=tsave,
        costs=costs,
        params_to_optimize=init_drive_params,
        filepath=filename,
        optimizer=optimizer,
        options=GRAPEOptions(progress_meter=None, epochs=4000),
        init_params_to_save=parser_args.__dict__,
    )

    finer_times = jnp.linspace(0.0, parser_args.time, 201)
    drive_spline = _drive_spline(opt_params["dp"], envelope, tsave)
    drive_amps = jnp.asarray(
        [drive_spline.evaluate(t) for t in finer_times]
    ).swapaxes(0, 1)
    fig, ax = plt.subplots()
    for drive_idx in range(len(H1s)):
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

    H = H_func(drive_params_dict=opt_params)
    plot_result = sesolve(
        H,
        initial_states,
        finer_times,
        exp_ops=[basis(2, idx) @ dag(basis(2, idx)) for idx in range(2)],
    )
    init_labels = [
        r'$|0\rangle$',
        r'$|1\rangle$',
        r'$|0\rangle+|1\rangle$',
        r'$|0\rangle+i|1\rangle$',
    ]
    exp_labels = [r'$|0\rangle$', r'$|1\rangle$']

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
