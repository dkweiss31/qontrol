import argparse
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import diffrax as dx

from dynamiqs import timecallable, dag, basis, destroy
from dynamiqs import sesolve

from opt_dynamiqs import grape, GRAPEOptions, generate_file_path, all_cardinal_states


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRAPE sim")
    parser.add_argument("--dim", default=4, type=int, help="tmon hilbert dim cutoff")
    parser.add_argument(
        "--Kerr", default=0.100, type=float, help="transmon Kerr in GHz"
    )
    parser.add_argument("--max_amp", default=[0.1, 0.1], help="max drive amp in GHz")
    parser.add_argument("--dt", default=2.0, type=float, help="time step for controls")
    parser.add_argument("--time", default=30.0, type=float, help="gate time")
    parser.add_argument(
        "--ramp_nts", default=3, type=int, help="numper of points in ramps"
    )
    parser.add_argument(
        "--scale",
        default=1e-5,
        type=float,
        help="randomization scale for initial pulse",
    )
    parser.add_argument(
        "--coherent", default=0, type=int, help="which fidelity metric to use"
    )
    parser.add_argument("--epochs", default=2000, type=int, help="number of epochs")
    parser.add_argument(
        "--target_fidelity", default=0.9995, type=float, help="target fidelity"
    )
    parser.add_argument(
        "--rng_seed", default=854, type=int, help="rng seed for random initial pulses"
    )  # 87336259
    parser.add_argument("--plot", default=True, type=bool, help="plot the results?")
    parser_args = parser.parse_args()
    filename = generate_file_path("h5py", "DRAG", "out")
    dim = parser_args.dim

    optimizer = optax.adam(learning_rate=0.0001, b1=0.99, b2=0.99)
    ntimes = int(parser_args.time // parser_args.dt) + 1
    tsave = jnp.linspace(0, parser_args.time, ntimes)

    # force the control endpoints to be at zero
    cos_ramp = (1 - jnp.cos(jnp.linspace(0.0, jnp.pi, parser_args.ramp_nts))) / 2
    envelope = jnp.concatenate(
        (cos_ramp, jnp.ones(ntimes - 2 * parser_args.ramp_nts), jnp.flip(cos_ramp))
    )
    options = GRAPEOptions(
        save_states=False,
        target_fidelity=parser_args.target_fidelity,
        epochs=parser_args.epochs,
        coherent=parser_args.coherent,
        progress_meter=None,
    )
    a = destroy(dim)
    H0 = -0.5 * parser_args.Kerr * 2.0 * jnp.pi * dag(a) @ dag(a) @ a @ a
    H1 = [a + dag(a), 1j * (a - dag(a))]
    H1_labels = ["I", "Q"]

    initial_states = [basis(dim, 0), basis(dim, 1)]
    final_states = [basis(dim, 1), basis(dim, 0)]

    # need to form superpositions so that the phase information is correct
    if not parser_args.coherent:
        initial_states = all_cardinal_states(initial_states)
        final_states = all_cardinal_states(final_states)

    rng = np.random.default_rng(parser_args.rng_seed)
    init_drive_params = (
        2.0
        * jnp.pi
        * (-2.0 * parser_args.scale * rng.random((len(H1), ntimes)) + parser_args.scale)
    )

    def _drive_spline(drive_params, envelope, ts):
        # note swap of axes so that time axis is first
        drive_w_envelope = jnp.einsum("t,dt->td", envelope, drive_params)
        drive_coeffs = dx.backward_hermite_coefficients(ts, drive_w_envelope)
        drive_spline = dx.CubicInterpolation(ts, drive_coeffs)
        return drive_spline

    def H_func(t, drive_params, envelope, ts):
        drive_spline = _drive_spline(drive_params, envelope, ts)
        drive_amps = drive_spline.evaluate(t)
        drive_Hs = jnp.einsum("d,dij->ij", drive_amps, H1)
        H = H0 + drive_Hs
        return H

    H_tc = jax.tree_util.Partial(H_func, envelope=envelope, ts=tsave)

    opt_params = grape(
        H_tc,
        initial_states=initial_states,
        target_states=final_states,
        tsave=tsave,
        params_to_optimize=init_drive_params,
        filepath=filename,
        optimizer=optimizer,
        options=options,
        init_params_to_save=parser_args.__dict__,
    )

    if parser_args.plot:
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
        ax.set_xlabel("time [ns]")
        ax.set_ylabel("pulse amplitude [GHz]")
        ax.legend()
        plt.tight_layout()
        plt.show()

        H_func = partial(H_func, drive_params=opt_params, envelope=envelope, ts=tsave)
        H_tc = timecallable(
            H_func,
        )
        plot_result = sesolve(
            H_tc,
            initial_states,
            finer_times,
            exp_ops=[basis(dim, idx) @ dag(basis(dim, idx)) for idx in range(dim)],
            options=options,
        )
        init_labels = [
            r"$|0\rangle$",
            r"$|0\rangle+|1\rangle$",
            r"$|0\rangle+i|1\rangle$",
            r"$|1\rangle$",
        ]
        exp_labels = [r"$|0\rangle$", r"$|1\rangle$", r"$|2\rangle$", r"$|3\rangle$"]

        for state_idx in range(len(initial_states)):
            fig, ax = plt.subplots()
            expects = plot_result.expects[state_idx]
            for e_result, label in zip(expects, exp_labels):
                plt.plot(finer_times, e_result, label=label)
            ax.legend()
            ax.set_xlabel("time [ns]")
            ax.set_ylabel("population")
            ax.set_title(f"initial state={init_labels[state_idx]}")
            plt.show()
