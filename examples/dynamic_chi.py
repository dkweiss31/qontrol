import argparse
from functools import partial

import diffrax as dx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import dynamiqs as dq
from dynamiqs import basis, dag, destroy, sesolve, timecallable, modulated, tensor, eye, unit
from dynamiqs import generate_noise_trajectory, pwc, mcsolve
from dynamiqs.solver import Tsit5
import jax
from jax.random import PRNGKey

from optamiqs import GRAPEOptions, generate_file_path, grape, PulseOptimizer, MCInfidelity
from optamiqs import IncoherentInfidelity, ForbiddenStates, all_cardinal_states, extract_info_from_h5
from optamiqs import infidelity_incoherent, write_to_h5
from cycler import cycler


dq.set_precision("double")

color_cycler = plt.rcParams['axes.prop_cycle']
ls_cycler = cycler(ls=['-', '--', '-.', ':'])
alpha_cycler = cycler(alpha=[1.0, 0.6, 0.2])
lw_cycler = cycler(lw=[2.0, 1.0])
color_ls_alpha_cycler = alpha_cycler * ls_cycler * color_cycler  # lw_cycler *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="dynamic chi sim")
    parser.add_argument("--idx", default=-1, type=int, help="idx to scan over")
    parser.add_argument("--gate", default="error_parity_plus_gf", type=str,
                        help="type of gate. Can be error_parity_g, error_parity_plus, ...")
    parser.add_argument("--drive_type", default="chi_gef", type=str,
                        help="type of drives. Can be chi_gf, chi_gef or gbs")
    parser.add_argument("--grape_type", default="mcsolve", type=str, help="can be sesolve, mesolve or mcsolve")
    parser.add_argument("--c_dim", default=3, type=int, help="cavity hilbert dim cutoff")
    parser.add_argument("--t_dim", default=3, type=int, help="tmon hilbert dim cutoff")
    parser.add_argument("--Kerr", default=0.100, type=float, help="transmon Kerr in GHz")
    parser.add_argument(
        "--max_amp",
        default=[0.002, 0.002, 0.01, 0.01],
        help="max drive amp in GHz"
    )
    parser.add_argument(
        "--scale",
        default=[0.0, 1e-2, 1e-2, 1e-2],
        help="randomization scale for initial pulse")
    parser.add_argument("--dt", default=40.0, type=float, help="time step for controls")
    parser.add_argument("--time", default=520.0, type=float, help="gate time")
    parser.add_argument("--ramp_nts", default=2, type=int, help="numper of points in ramps")
    parser.add_argument("--learning_rate", default=0.0001, type=float, help="learning rate for ADAM optimize")
    parser.add_argument("--b1", default=0.999, type=float, help="decay of learning rate first moment")
    parser.add_argument("--b2", default=0.999, type=float, help="decay of learning rate second moment")
    parser.add_argument("--coherent", default=0, type=int, help="which fidelity metric to use")
    parser.add_argument("--epochs", default=2000, type=int, help="number of epochs")
    parser.add_argument("--target_fidelity", default=0.990, type=float, help="target fidelity")
    parser.add_argument("--rng_seed", default=980, type=int, help="rng seed for random initial pulses")  # 87336259
    parser.add_argument("--include_low_frequency_noise", default=1, type=int,
                        help="whether to batch over different realizations of low-frequency noise")
    parser.add_argument("--num_freq_shift_trajs", default=3, type=int,
                        help="number of trajectories to sample low-frequency noise for")
    parser.add_argument("--sample_rate", default=1.0, type=float, help="rate at which to sample noise (in ns^-1)")
    parser.add_argument("--relative_PSD_strength", default=2e-6, type=float,
                        help="std-dev of frequency shifts given by sqrt(relative_PSD_strength * sample_rate)")
    parser.add_argument("--f0", default=1.0/100_000.0, type=float,
                        help="cutoff frequency for 1/f noise (in ns^-1), default is 1/100 us")
    parser.add_argument("--white", default=0, type=int, help="white or 1/f noise")
    parser.add_argument("--T1", default=10000, type=float, help="T1 of the transmon in ns")
    parser.add_argument("--ntraj", default=11, type=int, help="number of jump trajectories")
    parser.add_argument("--plot", default=True, type=bool, help="plot the results?")
    parser.add_argument("--plot_noise", default=True, type=bool, help="plot noise information")
    parser.add_argument("--initial_pulse_filepath",
                        default="out/00206_dynamic_chi_error_parity_plus_gf.h5py",
                        type=str, help="initial pulse filepath")
    parser.add_argument("--analysis_only", default=False, type=bool,
                        help="whether to actually run the grape optimization or "
                             "just analyze a pulse from initial_pulse_filepath")
    parser_args = parser.parse_args()
    if parser_args.idx == -1:
        filename = generate_file_path("h5py", f"dynamic_chi_{parser_args.gate}", "out")
    else:
        filename = f"out/{str(parser_args.idx).zfill(5)}_dynamic_chi_{parser_args.gate}.h5py"
    c_dim = parser_args.c_dim
    t_dim = parser_args.t_dim
    scale = jnp.asarray(parser_args.scale)

    optimizer = optax.adam(learning_rate=parser_args.learning_rate, b1=parser_args.b1, b2=parser_args.b2)
    ntimes = int(parser_args.time // parser_args.dt) + 1
    tsave = jnp.linspace(0, parser_args.time, ntimes)
    # force the control endpoints to be at zero
    begin_ramp = (1 - jnp.cos(jnp.linspace(0.0, jnp.pi, parser_args.ramp_nts))) / 2
    envelope = jnp.concatenate(
        (begin_ramp, jnp.ones(ntimes - 2 * parser_args.ramp_nts), jnp.flip(begin_ramp))
    )
    if parser_args.coherent == 0:
        coherent = False
    else:
        coherent = True
    options = GRAPEOptions(
        save_states=True,
        target_fidelity=parser_args.target_fidelity,
        epochs=parser_args.epochs,
        coherent=coherent,
        ntraj=parser_args.ntraj,
        grape_type=parser_args.grape_type,
        one_jump_only=True,
        progress_meter=None,
    )

    a = tensor(destroy(c_dim), eye(t_dim))
    b = tensor(eye(c_dim), destroy(t_dim))
    g_proj = tensor(eye(c_dim), basis(t_dim, 0) @ dag(basis(t_dim, 0)))
    e_proj = tensor(eye(c_dim), basis(t_dim, 1) @ dag(basis(t_dim, 1)))
    f_proj = tensor(eye(c_dim), basis(t_dim, 2) @ dag(basis(t_dim, 2)))
    gf_proj = tensor(eye(c_dim), basis(t_dim, 0) @ dag(basis(t_dim, 2)))
    ge_proj = tensor(eye(c_dim), basis(t_dim, 0) @ dag(basis(t_dim, 1)))
    ef_proj = tensor(eye(c_dim), basis(t_dim, 1) @ dag(basis(t_dim, 2)))
    if parser_args.drive_type == "chi_ge":
        H0 = 0.0 * b
        H1 = [dag(a) @ a @ f_proj, gf_proj + dag(gf_proj), 1j * (gf_proj - dag(gf_proj)), ]
        H1_labels = [r"$\chi_f$", r"$I_{gf}$", r"$Q_{gf}$"]
    elif parser_args.drive_type == "chi_gef":
        H0 = 0.0 * b
        H1 = [
            dag(a) @ a @ e_proj,
            dag(a) @ a @ f_proj,
            gf_proj + dag(gf_proj), 1j * (gf_proj - dag(gf_proj)),
        ]
        H1_labels = [
            r"$\chi_e$", r"$\chi_f$",
            r"$I_{gf}$", r"$Q_{gf}$",
        ]
    elif parser_args.drive_type == "gbs_delta":
        H0 = 0.0 * b
        H1 = [
            dag(a) @ b + a @ dag(b),
            1j * (dag(a) @ b - a @ dag(b)),
            gf_proj + dag(gf_proj),
            1j * (gf_proj - dag(gf_proj)),
        ]
        H1_labels = [
            r"$g_{\rm re}$", r"$\chi_f$",
            r"$I_{gf}$", r"$Q_{gf}$",
        ]
    else:
        raise ValueError(f"drive_type can be chi_ge, chi_gef,"
                         f" gbs but got {parser_args.drive_type}")
    assert len(H1) == len(H1_labels)
    if parser_args.grape_type == "mcsolve":
        jump_ops = [jnp.sqrt(1. / parser_args.T1) * b, ]
    else:
        jump_ops = None
    if type(parser_args.max_amp) is float:
        max_amp = len(H1) * [2.0 * jnp.pi * parser_args.max_amp]
    elif len(parser_args.max_amp) == len(H1):
        max_amp = 2.0 * jnp.pi * jnp.asarray(parser_args.max_amp)
    else:
        raise RuntimeError("max_amp needs to be a float or have the same dimension as H1")
    if parser_args.gate == "error_parity_g":
        initial_states = [tensor(basis(c_dim, c_idx), basis(t_dim, 0))
                          for c_idx in range(2)]
        final_states = [tensor(basis(c_dim, c_idx), basis(t_dim, c_idx % 2))
                        for c_idx in range(2)]
        final_states_traj = None
    elif parser_args.gate == "error_parity_plus":
        initial_states = [tensor(basis(c_dim, c_idx), unit(basis(t_dim, 0) + basis(t_dim, 1)))
                          for c_idx in range(2)]
        final_states = [tensor(basis(c_dim, 0), unit(basis(t_dim, 0) + basis(t_dim, 1))),
                        1j * tensor(basis(c_dim, 1), unit(basis(t_dim, 0) - basis(t_dim, 1)))]
        final_states_traj = None
    elif parser_args.gate == "error_parity_plus_gf":
        initial_states = [tensor(basis(c_dim, c_idx), unit(basis(t_dim, 0) + basis(t_dim, 2)))
                          for c_idx in range(2)]
        final_states = [tensor(basis(c_dim, 0), unit(basis(t_dim, 0) + basis(t_dim, 2))),
                        1j * tensor(basis(c_dim, 1), unit(basis(t_dim, 0) - basis(t_dim, 2)))]
        final_states_traj = [
            tensor(basis(c_dim, 0), basis(t_dim, 1)),
            -1j * tensor(basis(c_dim, 1), basis(t_dim, 1))
        ]
    # elif parser_args.gate == "e_swap_pi_4":
    else:
        raise RuntimeError("gate type not supported")

    if parser_args.coherent == 0:
        # pass
        initial_states = all_cardinal_states(initial_states)
        final_states = all_cardinal_states(final_states)
        if final_states_traj is not None:
            final_states_traj = all_cardinal_states(final_states_traj)

    if parser_args.include_low_frequency_noise:
        delay_times = jnp.linspace(0.0, 2000, 501)
        noise_time = np.max([delay_times[-1], parser_args.time])
        noise_t_list, noise_shifts, traj, freq_list, psd = generate_noise_trajectory(
            3 * parser_args.num_freq_shift_trajs, parser_args.sample_rate,
            noise_time, parser_args.relative_PSD_strength,
            parser_args.f0, parser_args.white, parser_args.rng_seed
        )
        noise_shifts = jnp.reshape(noise_shifts, (len(noise_t_list), 3, parser_args.num_freq_shift_trajs))
        noise_shifts = jnp.transpose(noise_shifts, (1, 0, 2))
        noise_coeffs = [
            dx.backward_hermite_coefficients(noise_t_list, noise_shift)
            for noise_shift in noise_shifts
        ]
        noise_splines = [dx.CubicInterpolation(noise_t_list, noise_coeff) for noise_coeff in noise_coeffs]
        finer_times = jnp.linspace(0.0, parser_args.time, 101)
        if parser_args.plot_noise:
            psd = jnp.mean(psd, axis=-1)
            std_dev_trajectory = np.std(traj, axis=-1)
            fig, ax = plt.subplots()
            plt.loglog(freq_list, psd)
            plt.ylabel(r"Power spectral density [ns$^{-1}$]")
            plt.xlabel("Noise frequency [Ghz]")
            plt.grid()
            plt.tight_layout()
            plt.show()

            fig, ax = plt.subplots()
            for idx in range(parser_args.num_freq_shift_trajs):
                plt.plot(noise_t_list, traj[:, idx])
            plt.plot(noise_t_list, np.sqrt(noise_t_list * parser_args.relative_PSD_strength), 'k--', label=r'Expected $\sqrt{PSD}$')
            plt.plot(noise_t_list, -np.sqrt(noise_t_list * parser_args.relative_PSD_strength), 'k--')
            plt.plot(noise_t_list, std_dev_trajectory, 'k', label=r'Trajectory std. dev.')
            plt.plot(noise_t_list, -std_dev_trajectory, 'k')
            plt.title('1/f noise trajectories', pad=10)
            plt.xlabel(r"Time [ns]", labelpad=12)
            plt.ylabel(r"Phase shift / $2\pi$", labelpad=12)
            plt.legend(loc='upper left')
            plt.tight_layout()
            plt.show()

            fig, ax = plt.subplots()
            for idx, sty in zip(range(parser_args.num_freq_shift_trajs), color_ls_alpha_cycler):
                noise_amp = jnp.asarray([noise_splines[idx].evaluate(t) for t in finer_times])
                plt.plot(finer_times, 10 ** 3 * noise_amp, lw=0.6)
            plt.xlabel("time [ns]")
            plt.ylabel("amplitude [MHz]")
            plt.tight_layout()
            plt.show()

            def proj_for_T2(idx):
                return basis(3, idx) @ dag(basis(3, idx))
            g_proj_noise, e_proj_noise, f_proj_noise = proj_for_T2(0), proj_for_T2(1), proj_for_T2(2)

            def H_noise_func(t):
                noise_amps = jnp.stack([noise_spline.evaluate(t) for noise_spline in noise_splines])
                return jnp.einsum(
                    "sb,sjk->bjk",
                    2.0 * jnp.pi * noise_amps,
                    jnp.asarray([g_proj_noise, e_proj_noise, f_proj_noise]),
                )
            # H_noise_tc = timecallable(H_noise_func)
            # init_state = unit(basis(3, 0) + basis(3, 2))
            # readout_proj = init_state @ dag(init_state)
            # T2_Ramsey_probs = T2_Ramsey_experiment(
            #     H_noise_tc,
            #     init_state,
            #     delay_times,
            #     readout_proj,
            # )
            # T_phi_ramsey_gauss = extract_Tphi(
            #     T2_Ramsey_probs, delay_times, type="gauss", plot=parser_args.plot_noise
            # )
            # X_op_half = basis(3, 0) @ dag(basis(3, 2))
            # X_op = X_op_half + dag(X_op_half)
            # T2_echo_probs = T2_echo_experiment(
            #     H_noise_tc,
            #     init_state,
            #     delay_times,
            #     X_op,
            #     readout_proj,
            # )
            # # gamma_phi_ramsey_exp = extract_Tphi(
            # #     T2_Ramsey_probs, delay_times, type="exp"
            # # )
            # T_phi_echo_gauss = extract_Tphi(
            #     T2_echo_probs, delay_times, type="gauss", plot=parser_args.plot_noise
            # )
            # T_phi_echo_exp = extract_Tphi(
            #     T2_echo_probs, delay_times, type="exp", plot=parser_args.plot_noise
            # )
            # # gamma_phi_echo = extract_gammaphi(T2_echo_probs, delay_times)
            # # print(gamma_phi_echo, gamma_phi_ramsey)
            # print("Ramsey time is: ", T_phi_ramsey_gauss, "us")
            # print("Echo time (gauss) is: ", T_phi_echo_gauss, "us")
            # print("Echo time (exp) is: ", T_phi_echo_exp, "us")

    rng = np.random.default_rng(parser_args.rng_seed)

    if parser_args.initial_pulse_filepath is not None:
        data, params = extract_info_from_h5(parser_args.initial_pulse_filepath)
        init_drive_params = data["opt_params"][-1]
    else:
        init_drive_params = jnp.einsum(
            "h,ht->ht",
            2.0 * jnp.pi * (-2.0) * scale,
            rng.random((len(H1), ntimes))
        )
        init_drive_params += 2.0 * jnp.pi * scale[:, None]


    def _drive_spline(drive_params, envelope, ts, max_amp):
        # note swap of axes so that time axis is first
        drive_w_envelope = jnp.einsum("t,...t->t...", envelope, drive_params)
        total_drive = jnp.clip(
            drive_w_envelope,
            a_min=-max_amp,
            a_max=max_amp,
        )
        drive_coeffs = dx.backward_hermite_coefficients(ts, total_drive)
        drive_spline = dx.CubicInterpolation(ts, drive_coeffs)
        return drive_spline

    def H_func(drive_params, envelope, ts):
        H = H0
        for drive_idx, (_H1, drive_param) in enumerate(zip(H1, drive_params)):
            drive_spline = _drive_spline(drive_param, envelope, ts, max_amp=max_amp[drive_idx])
            H += modulated(drive_spline.evaluate, _H1)
        if parser_args.include_low_frequency_noise:
            # extra factor of 2 is because Aniket defines it as 2 pi sigmaz
            noise_ops = jnp.stack([g_proj, e_proj, f_proj])
            for (_H1, noise_spline) in zip(noise_ops, noise_splines):
                H += modulated(
                    lambda t: 2.0 * jnp.pi * noise_spline.evaluate(t), _H1
                )
        return H

    #####
    # operators for plotting
    ######
    if parser_args.gate == "error_parity_plus_gf":
        e_idx = 2
    else:
        e_idx = 1

    X_ops = [
        tensor(basis(c_dim, c_idx), basis(t_dim, 0)) @ dag(tensor(basis(c_dim, c_idx), basis(t_dim, e_idx)))
        + tensor(basis(c_dim, c_idx), basis(t_dim, e_idx)) @ dag(tensor(basis(c_dim, c_idx), basis(t_dim, 0)))
        for c_idx in range(c_dim)
    ]
    X_labels = [f"X_{c_idx}" for c_idx in range(c_dim)]
    Y_ops = [
        1j * tensor(basis(c_dim, c_idx), basis(t_dim, 0)) @ dag(tensor(basis(c_dim, c_idx), basis(t_dim, e_idx)))
        - 1j * tensor(basis(c_dim, c_idx), basis(t_dim, e_idx)) @ dag(tensor(basis(c_dim, c_idx), basis(t_dim, 0)))
        for c_idx in range(c_dim)
    ]
    Y_labels = [f"Y_{c_idx}" for c_idx in range(c_dim)]
    Z_ops = [
        tensor(basis(c_dim, c_idx), basis(t_dim, 0)) @ dag(tensor(basis(c_dim, c_idx), basis(t_dim, 0)))
        - tensor(basis(c_dim, c_idx), basis(t_dim, e_idx)) @ dag(tensor(basis(c_dim, c_idx), basis(t_dim, e_idx)))
        for c_idx in range(c_dim)
    ]
    exp_ops = X_ops + Y_ops + Z_ops
    Z_labels = [f"Z_{c_idx}" for c_idx in range(c_dim)]
    labels = X_labels + Y_labels + Z_labels

    #####
    ntimes_fixed_chi = 1001
    half_idx = ntimes_fixed_chi // 2
    tsave_fixed_chi = jnp.linspace(0, parser_args.time, ntimes_fixed_chi)
    zero_drive = jnp.zeros(ntimes_fixed_chi)
    fixed_chi = (np.pi / (tsave_fixed_chi[-1])) * jnp.ones(ntimes_fixed_chi)
    echoed_chi = jnp.where(
        tsave_fixed_chi[half_idx] > tsave_fixed_chi, fixed_chi, -fixed_chi
    )
    n_dt = 6
    pi_dt = tsave_fixed_chi[n_dt] - tsave_fixed_chi[0]
    pi_amp = jnp.pi / (2 * pi_dt)
    pi_pulse = jnp.zeros(ntimes_fixed_chi - 1)
    for idx in range(n_dt):
        pi_pulse = pi_pulse.at[half_idx - n_dt//2 + idx].set(pi_amp)
    H_qubit_pi2 = pwc(
        tsave_fixed_chi,
        pi_pulse,
        gf_proj + dag(gf_proj),
    )

    drive_params_fixed_chi = jnp.vstack((
        # zero_drive,
        zero_drive,
        fixed_chi,
        zero_drive,
        zero_drive,
    ))
    drive_params_echoed_chi = jnp.vstack((
        # zero_drive,
        zero_drive,
        echoed_chi,
        zero_drive,
        zero_drive,
    ))
    envelope_fixed_chi = jnp.concatenate(
        (begin_ramp, jnp.ones(ntimes_fixed_chi - 2 * parser_args.ramp_nts), jnp.flip(begin_ramp))
    )

    final_states_fixed = [tensor(basis(c_dim, 0), unit(basis(t_dim, 0) + basis(t_dim, e_idx))),
                          tensor(basis(c_dim, 1), unit(basis(t_dim, 0) - basis(t_dim, e_idx)))]
    final_states_traj_fixed = [tensor(basis(c_dim, 0), basis(t_dim, 1)),
                               -1 * tensor(basis(c_dim, 1), basis(t_dim, 1))]
    final_states_echo = [tensor(basis(c_dim, 0), unit(basis(t_dim, 0) + basis(t_dim, 2))),
                         - 1j * tensor(basis(c_dim, 1), unit(basis(t_dim, 0) - basis(t_dim, 2)))]
    final_states_traj_echo = [tensor(basis(c_dim, 0), basis(t_dim, 1)),
                              + 1j * tensor(basis(c_dim, 1), basis(t_dim, 1))]
    if parser_args.coherent == 0:
        final_states_fixed = all_cardinal_states(final_states_fixed)
        final_states_traj_fixed = all_cardinal_states(final_states_traj_fixed)
        final_states_echo = all_cardinal_states(final_states_echo)
        final_states_traj_echo = all_cardinal_states(final_states_traj_echo)

    H_func_fixed = H_func(
        drive_params=drive_params_fixed_chi,
        envelope=envelope_fixed_chi,
        ts=tsave_fixed_chi
    )

    _H_func_echoed = H_func(
        drive_params=drive_params_echoed_chi,
        envelope=envelope_fixed_chi,
        ts=tsave_fixed_chi,
    )

    H_func_echoed = _H_func_echoed + H_qubit_pi2

    result_fixed = sesolve(
        H_func_fixed, initial_states, tsave_fixed_chi, exp_ops=X_ops, options=options
    )
    result_echoed = sesolve(
        H_func_echoed, initial_states, tsave_fixed_chi, exp_ops=X_ops, options=options,
        solver=Tsit5(max_steps=1_000_000)
    )
    infid_fixed_chi = infidelity_incoherent(
        result_fixed.final_state, jnp.asarray(final_states_fixed)
    )
    infid_echoed_chi = infidelity_incoherent(
        result_echoed.final_state, jnp.asarray(final_states_echo)
    )
    print(f"coherent fidelity for the fixed chi pulse is {1 - np.average(infid_fixed_chi)}")
    print(f"coherent fidelity for the echoed chi pulse is {1 - np.average(infid_echoed_chi)}")
    infid_dict = {
        "infid_coherent_fixed": np.average(infid_fixed_chi),
        "infid_coherent_echo": np.average(infid_echoed_chi),
    }

    def mcsolve_infids(_result, _final_states, _final_states_traj):
        _final_jump_states = unit(_result.final_jump_states).swapaxes(-4, -3)
        _final_no_jump_states = unit(_result.final_no_jump_state)

        def _average_over_batches(infid_array):
            return jnp.average(infid_array, axis=range(len(infid_array.shape) - 1))
        _infids_jump = infidelity_incoherent(
            _final_jump_states, jnp.asarray(_final_states_traj), average=False
        )
        _infids_no_jump = infidelity_incoherent(
            _final_no_jump_states, jnp.asarray(_final_states), average=False
        )
        _infids_jump = _average_over_batches(_infids_jump)
        _infids_no_jump = _average_over_batches(_infids_no_jump)
        _p_nojump = _average_over_batches(_result.no_jump_prob)
        _infid = _p_nojump * _infids_no_jump + (1 - _p_nojump) * _infids_jump
        return _p_nojump, _infid, _infids_no_jump, _infids_jump

    if parser_args.grape_type == "mcsolve":
        mc_result_fixed = mcsolve(
            H_func_fixed, jump_ops, initial_states, tsave_fixed_chi, options=options
        )
        mc_result_echoed = mcsolve(
            H_func_echoed, jump_ops, initial_states, tsave_fixed_chi, options=options
        )

        (p_nojump_fixed,
         infid_fixed,
         infids_no_jump_fixed,
         infids_jump_fixed) = mcsolve_infids(
            mc_result_fixed, final_states_fixed, final_states_traj_fixed
        )
        (p_nojump_echo,
         infid_echo,
         infids_no_jump_echo,
         infids_jump_echo) = mcsolve_infids(
            mc_result_echoed, final_states_echo, final_states_traj_echo
        )
        print("jump infidelities for fixed and echo are ",
              np.average(infids_jump_fixed), np.average(infids_jump_echo))
        print("Weighted fidelities for fixed and echo are ",
              1 - np.average(infid_fixed), 1 - np.average(infid_echo))
        infid_dict["p_nojump_fixed"] = np.average(p_nojump_fixed)
        infid_dict["infid_fixed"] = np.average(infid_fixed)
        infid_dict["infid_no_jump_fixed"] = np.average(infids_no_jump_fixed)
        infid_dict["infid_jump_fixed"] = np.average(infids_jump_fixed)

        infid_dict["p_nojump_echo"] = np.average(p_nojump_echo)
        infid_dict["infid_echo"] = np.average(infid_echo)
        infid_dict["infid_no_jump_echo"] = np.average(infids_no_jump_echo)
        infid_dict["infid_jump_echo"] = np.average(infids_jump_echo)

    H_tc = jax.tree_util.Partial(H_func, envelope=envelope, ts=tsave)

    pulse_optimizer = PulseOptimizer(H_tc, lambda _H, _dp: (_H(_dp), tsave))

    costs = [MCInfidelity(target_states=final_states, target_states_traj=final_states_traj,
                          coherent=coherent, no_jump_weight=1.0, jump_weight=1.0), ]

    if not parser_args.analysis_only:
        opt_params = grape(
            pulse_optimizer,
            initial_states=initial_states,
            costs=costs,
            jump_ops=jump_ops,
            params_to_optimize=init_drive_params,
            filepath=filename,
            optimizer=optimizer,
            options=options,
            init_params_to_save=parser_args.__dict__,
        )
    else:
        opt_params = init_drive_params

    if parser_args.plot:

        finer_times = jnp.linspace(0.0, parser_args.time, 201)
        drive_splines = [_drive_spline(opt_param, envelope, tsave, max_amp[idx])
                         for idx, opt_param in enumerate(opt_params)]
        # init_drive_spline = _drive_spline(init_drive_params, envelope, tsave)
        drive_amps = [[drive_spline.evaluate(t) for t in finer_times]
                      for drive_spline in drive_splines]
        # init_drive_amps = jnp.asarray([init_drive_spline.evaluate(t) for t in finer_times]).swapaxes(0, 1)

        fig, ax = plt.subplots()
        for drive_idx in range(len(H1)):
            plt.plot(
                finer_times,
                jnp.asarray(drive_amps[drive_idx])/(2.0*np.pi),
                label=H1_labels[drive_idx]
            )
        # plt.plot(finer_times, (np.pi / (2.0 * np.pi * tsave[-1])) * jnp.ones_like(finer_times),
        #          ls="--", color="black", label="chi")
        # plt.plot(finer_times, (-np.pi / (2.0 * np.pi * tsave[-1])) * jnp.ones_like(finer_times),
        #          ls="--", color="black")
        ax.set_xlabel("time [ns]")
        ax.set_ylabel("pulse amplitude [GHz]")
        ax.set_title(filename)
        ax.legend()
        plt.tight_layout()
        plt.savefig(filename[:-5]+"_pulse.pdf")
        plt.show()

        def Pij(c_idx, t_idx):
            ket = tensor(basis(c_dim, c_idx), basis(t_dim, t_idx))
            return ket @ dag(ket)


        H_tc = H_func(drive_params=opt_params, envelope=envelope, ts=tsave)

        if parser_args.grape_type == "mcsolve":
            opt_result = mcsolve(H_tc, jump_ops, initial_states, tsave,
                                 key=PRNGKey(parser_args.rng_seed), options=options)
            plot_result = sesolve(H_tc, initial_states, finer_times,
                                  exp_ops=exp_ops, options=options)
            (p_nojump,
             infid,
             infids_no_jump,
             infids_jump) = mcsolve_infids(opt_result, final_states, final_states_traj)
            print(f"final average jump fidelity is {1-jnp.mean(infids_jump)}")
            print(f"final average no-jump fidelity is {1-jnp.mean(infids_no_jump)}")
            print(f"final weighted fidelity is {1 - jnp.mean(infid)}")
            infid_dict = infid_dict | {
                "p_nojump": np.average(p_nojump),
                "infid_jump": np.average(infids_jump),
                "infid_no_jump": np.average(infids_no_jump),
                "infid": np.average(infid),
            }
        else:
            plot_result = sesolve(H_tc, initial_states, finer_times, exp_ops=exp_ops)
            infid = infidelity_incoherent(
                plot_result.final_state, jnp.asarray(final_states)
            )
            infid_dict = infid_dict | {
                "infid_sesolve": np.average(infid),
            }
            print(f"final fidelity is {1-np.average(infid)}")
        infid_filename = generate_file_path("h5py", f"analysis_infid", "out")
        if parser_args.plot_noise and parser_args.include_low_frequency_noise:
            # dephasing_times_dict = {
            #     "T_phi_ramsey_gauss": T_phi_ramsey_gauss,
            #     "T_phi_echo_exp": T_phi_echo_exp,
            #     "T_phi_echo_gauss": T_phi_echo_gauss,
            # }
            infid_dict = infid_dict  # | dephasing_times_dict
        print(f"writing infid data to {infid_filename}")
        write_to_h5(infid_filename, infid_dict, parser_args.__dict__)

        #####
        # plot the fixed and echoed chi results
        # initial state without photon in cavity
        fig, ax = plt.subplots()
        for idx in range(parser_args.num_freq_shift_trajs):
            plt.plot(tsave_fixed_chi, result_fixed.expects[idx, 0, 0, :], ls="--")
            plt.plot(tsave_fixed_chi, result_echoed.expects[idx, 0, 0, :], ls="-")
            plt.plot(finer_times, plot_result.expects[idx, 0, 0, :], ls="-.")
        # ax.legend()
        ax.set_title(infid_filename + "\n" + "cavity in 0" + "\n" + "solid: echo, dashed: fixed, dash-dot: OCT")
        ax.set_ylabel(r"$\langle X \rangle$")
        ax.set_xlabel("time [ns]")
        plt.show()
        # initial state with photon in cavity
        fig, ax = plt.subplots()
        for idx in range(parser_args.num_freq_shift_trajs):
            plt.plot(tsave_fixed_chi, result_fixed.expects[idx, 3, 1, :], ls="--")
            plt.plot(tsave_fixed_chi, result_echoed.expects[idx, 3, 1, :], ls="-")
            plt.plot(finer_times, plot_result.expects[idx, 3, 1, :], ls="-.")
        # ax.legend()
        ax.set_title(infid_filename + "\n" + "cavity in 1" + "\n" + "solid: echo, dashed: fixed, dash-dot: OCT")
        ax.set_ylabel(r"$\langle X \rangle$")
        ax.set_xlabel("time [ns]")
        plt.show()

        for state_idx in range(len(initial_states)):
            fig, ax = plt.subplots()
            # plot the first noisy trajectory
            if parser_args.include_low_frequency_noise:
                expects = plot_result.expects[0][state_idx]
            else:
                expects = plot_result.expects[state_idx]
            for e_result, label, sty in zip(expects, labels, color_ls_alpha_cycler):
                plt.plot(finer_times, e_result, label=label, **sty)
            ax.legend()
            ax.set_xlabel("time [ns]")
            ax.set_ylabel("population")
            ax.set_title(f"state_idx={state_idx}")
            plt.show()
