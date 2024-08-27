import argparse

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from dynamiqs import basis, dag, destroy, pwc, sesolve
from dynamiqs.solver import Tsit5

from optamiqs import GRAPEOptions, all_cardinal_states, generate_file_path, grape, PulseOptimizer
from optamiqs import IncoherentInfidelity

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GRAPE sim')
    parser.add_argument('--dim', default=4, type=int, help='tmon hilbert dim cutoff')
    parser.add_argument(
        '--Kerr', default=0.100, type=float, help='transmon Kerr in GHz'
    )
    parser.add_argument('--dt', default=2.0, type=float, help='time step for controls')
    parser.add_argument('--time', default=30.0, type=float, help='gate time')
    parser_args = parser.parse_args()
    filename = generate_file_path('h5py', 'Kerr_pwc', 'out')
    dim = parser_args.dim

    optimizer = optax.adam(learning_rate=0.0001, b1=0.99, b2=0.99)
    ntimes = int(parser_args.time // parser_args.dt) + 1
    tsave = jnp.linspace(0, parser_args.time, ntimes)

    # TODO not working now because of the pwc bug for save_states=True
    options = GRAPEOptions(
        save_states=True,
        progress_meter=None,
    )
    a = destroy(dim)
    H0 = -0.5 * parser_args.Kerr * 2.0 * jnp.pi * dag(a) @ dag(a) @ a @ a
    H1 = [a + dag(a), 1j * (a - dag(a))]
    H1_labels = ['I', 'Q']

    initial_states = [basis(dim, 0), basis(dim, 1)]
    final_states = [basis(dim, 1), basis(dim, 0)]

    # need to form superpositions so that the phase information is correct
    if not options.coherent:
        initial_states = all_cardinal_states(initial_states)
        final_states = all_cardinal_states(final_states)

    # initial guess for pwc pulses
    init_drive_params = -0.001 * jnp.ones((len(H1), ntimes - 1))

    # defining the Hamiltonian as H0 plus pwc pulses
    def H_pwc(values):
        H = H0
        for idx, _H1 in enumerate(H1):
            H += pwc(tsave, values[idx], _H1)
        return H

    pulse_optimizer = PulseOptimizer(
        H_pwc, update_function=lambda _H, _dp: (_H(_dp), tsave)
    )
    costs = [IncoherentInfidelity(target_states=final_states, cost_multiplier=1.0), ]

    opt_params = grape(
        pulse_optimizer,
        initial_states=initial_states,
        costs=costs,
        params_to_optimize=init_drive_params,
        filepath=filename,
        optimizer=optimizer,
        options=options,
        solver=Tsit5(max_steps=1_000_000_000),
        init_params_to_save=parser_args.__dict__,
    )

    # plot the pulse
    finer_times = jnp.linspace(0.0, parser_args.time, 201)
    opt_H1s = [pwc(tsave, opt_params[drive_idx]/(2.0 * np.pi), _H1)
               for drive_idx, _H1 in enumerate(H1)]
    fig, ax = plt.subplots()
    for drive_idx in range(len(H1)):
        plt.plot(
            finer_times,
            [opt_H1s[drive_idx].prefactor(t) for t in finer_times],
            label=H1_labels[drive_idx],
        )
    ax.set_xlabel('time [ns]')
    ax.set_ylabel('pulse amplitude [GHz]')
    ax.legend()
    plt.tight_layout()
    plt.show()
