import diffrax as dx
import jax.numpy as jnp
import optax
from dynamiqs import mesolve, modulated, sesolve
from jax import Array

import qontrol
from qontrol import hamiltonian_time_updater


class AbstractSystem:
    def __init__(
        self, H0, H1s, jump_ops, initial_states, target_states, tsave, options
    ):
        self.H0 = H0
        self.H1s = H1s
        self.jump_ops = jump_ops
        self.initial_states = initial_states
        self.target_states = target_states
        self.tsave = tsave
        self.options = options

    def assert_correctness(self, opt_params, H_t_updater, infid_cost):
        opt_H, tsave = H_t_updater.update(opt_params)
        if self.options.grape_type == 0:  # sesolve
            opt_result = sesolve(opt_H, self.initial_states, tsave)
        else:  # mesolve
            opt_result = mesolve(opt_H, self.jump_ops, self.initial_states, tsave)
        infid = infid_cost.evaluate(opt_result, opt_H)
        assert infid < 1 - self.options.target_fidelity

    def run(self, costs, filepath):
        init_drive_params = {'dp': -0.001 * jnp.ones((len(self.H1s), len(self.tsave)))}

        def _drive_spline(drive_params: Array) -> dx.CubicInterpolation:
            drive_coeffs = dx.backward_hermite_coefficients(self.tsave, drive_params)
            return dx.CubicInterpolation(self.tsave, drive_coeffs)

        def H_func(drive_params_dict: dict) -> Array:
            drive_params = drive_params_dict['dp']
            H = self.H0
            for H1, drive_param in zip(self.H1s, drive_params):
                drive_spline = _drive_spline(drive_param)
                H += modulated(drive_spline.evaluate, H1)
            return H

        def update_function(H, drive_params_dict):
            new_H = H(drive_params_dict)
            return new_H, self.tsave

        H_t_updater = hamiltonian_time_updater(H_func, update_function)
        optimizer = optax.adam(0.0001, b1=0.99, b2=0.99)

        opt_params = qontrol.grape(
            H_t_updater,
            self.initial_states,
            init_drive_params,
            costs=costs,
            jump_ops=self.jump_ops,
            options=self.options,
            filepath=filepath,
            optimizer=optimizer,
        )

        return opt_params, H_t_updater
