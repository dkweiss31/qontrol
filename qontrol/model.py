from __future__ import annotations

import dynamiqs as dq
import equinox as eqx
import jax.numpy as jnp
import jax.random
import jax.tree_util as jtu
import optimistix as optx
from dynamiqs import TimeArray
from dynamiqs._utils import cdtype
from dynamiqs.gradient import Gradient
from dynamiqs.integrators._utils import _astimearray
from dynamiqs.result import Result
from dynamiqs.solver import Solver, Tsit5
from jax import Array
from jaxtyping import ArrayLike

from .options import OptimizerOptions


def sesolve_model(
    H_function: callable,
    psi0: ArrayLike,
    tsave_or_function: ArrayLike | callable,
    *,
    exp_ops: list[ArrayLike] | None = None,
) -> SESolveModel:
    r"""Instantiate sesolve model.

    Here we instantiate the model that is called at each step of the optimization
    iteration, returning a tuple of the result of calling `sesolve` as well as the
    Hamiltonian evaluated at the parameter values.

    Args:
        H_function _(callable)_: function specifying how to update the Hamiltonian
        psi0 _(ArrayLike of shape (..., n, 1))_: Initial states.
        tsave_or_function _(ArrayLike of shape (ntsave,) or callable)_: Either an
            array of times passed to sesolve or a method specifying how to update
            the times that are passed to sesolve
        exp_ops _(list of array-like)_: Operators to calculate expectation values of,
            in case some of the cost functions depend on the value of certain
            expectation values.

    Returns:
        _(SESolveModel)_: Model that when called with the parameters we optimize
            over as argument returns the results of `sesolve` as well as the updated
            Hamiltonian

    Examples:
        In this simple example the parameters are the amplitudes of piecewise-constant
        controls
        ```python
        tsave = jnp.linspace(0.0, 11.0, 10)
        psi0 = [dq.basis(2, 0)]
        H1s = [dq.sigmax(), dq.sigmay()]


        def H_pwc(values: Array) -> dq.TimeArray:
            H = dq.sigmaz()
            for idx, _H1 in enumerate(H1s):
                H += dq.pwc(tsave, values[idx], _H1)
            return H


        sesolve_model = ql.sesolve_model(H_pwc, psi0, tsave)
        ```
        See for example [this tutorial](../examples/qubit).

        In more complex cases, we can imagine that the optimized parameters
        are the control points fed into a spline, and additionally the control
        times themselves are optimized.
        ```python
        init_drive_params_topt = {
            'dp': -0.001 * jnp.ones((len(H1s), len(tsave))),
            't': tsave[-1],
        }


        def H_func_topt(t: float, drive_params_dict: dict) -> dq.TimeArray:
            drive_params = drive_params_dict['dp']
            new_tsave = jnp.linspace(0.0, drive_params_dict['t'], len(tsave))
            drive_spline = _drive_spline(drive_params, envelope, new_tsave)
            drive_amps = drive_spline.evaluate(t)
            drive_Hs = jnp.einsum('d,dij->ij', drive_amps, H1s)
            return H0 + drive_Hs


        def update_H_topt(drive_params_dict: dict) -> dq.TimeArray:
            new_H = jtu.Partial(H_func_topt, drive_params_dict=drive_params_dict)
            return dq.timecallable(new_H)


        def update_tsave_topt(drive_params_dict: dict) -> jax.Array:
            return jnp.linspace(0.0, drive_params_dict['t'], len(tsave))


        se_t_opt_Kerr_model = ql.sesolve_model(update_H_topt, psi0, update_tsave_topt)
        ```
        See for example [this tutorial](../examples/Kerr_oscillator#time-optimal-control).

    """  # noqa E501
    H_function, psi0, tsave_or_function, exp_ops = _initialize_model(
        H_function, psi0, tsave_or_function, exp_ops
    )
    return SESolveModel(H_function, psi0, tsave_or_function, exp_ops)


def mesolve_model(
    H_function: callable,
    jump_ops: list[ArrayLike],
    rho0: ArrayLike,
    tsave_or_function: ArrayLike | callable,
    *,
    exp_ops: list[ArrayLike] | None = None,
) -> MESolveModel:
    r"""Instantiate mesolve model.

    Here we instantiate the model that is called at each step of the optimization
    iteration, returning a tuple of the result of calling `mesolve` as well as the
    Hamiltonian evaluated at the parameter values.

    Args:
        H_function _(callable)_: function specifying how to update the Hamiltonian
        jump_ops _(list of array-like or time-array, each of shape (...Lk, n, n))_:
            List of jump operators.
        rho0 _(ArrayLike of shape (..., n, n))_: Initial density matrices.
        tsave_or_function _(ArrayLike of shape (ntsave,) or callable)_: Either an
            array of times passed to sesolve or a method specifying how to update
            the times that are passed to sesolve
        exp_ops _(list of array-like)_: Operators to calculate expectation values of,
            in case some of the cost functions depend on the value of certain
            expectation values.

    Returns:
        _(MESolveModel)_: Model that when called with the parameters we optimize
            over as argument returns the results of `mesolve` as well as the updated
            Hamiltonian

    Examples:
        Instantiating a `MESolveModel` is quite similar to instantiating an
        `SESolveModel`, with the two differences being that we need to supply jump
        operators, and the initial and target states should be specified as density
        matrices. Continuing the last example from `sesolve_model`
        ```python
        jump_ops = [0.03 * dq.sigmax()]
        me_initial_states = dq.todm(psi0)
        me_Kerr_model = ql.mesolve_model(
            update_H_topt, jump_ops, me_initial_states, update_tsave_topt
        )
        ```
        See [this tutorial](../examples/Kerr_oscillator#master-equation-optimization)
        for example
    """
    H_function, rho0, tsave_function, exp_ops = _initialize_model(
        H_function, rho0, tsave_or_function, exp_ops
    )
    jump_ops = [_astimearray(L) for L in jump_ops]
    return MESolveModel(H_function, rho0, tsave_function, exp_ops, jump_ops)


def mcsolve_model(
    H_function: callable,
    jump_ops: list[ArrayLike],
    psi0: ArrayLike,
    tsave_or_function: ArrayLike | callable,
    *,
    exp_ops: list[ArrayLike] | None = None,
    keys: Array = jax.random.split(jax.random.key(31), num=10),  # noqa B008
) -> MCSolveModel:
    r"""Instantiate mcsolve model.

    Here we instantiate the model that is called at each step of the optimization
    iteration, returning a tuple of the result of calling `mcsolve` as well as the
    Hamiltonian evaluated at the parameter values.

    Args:
        H_function _(callable)_: function specifying how to update the Hamiltonian
        jump_ops _(list of array-like or time-array, each of shape (...Lk, n, n))_:
            List of jump operators.
        psi0 _(ArrayLike of shape (..., n, 1))_: Initial states.
        tsave_or_function _(ArrayLike of shape (ntsave,) or callable)_: Either an
            array of times passed to sesolve or a method specifying how to update
            the times that are passed to sesolve
        keys _(ArrayLike of shape (ntraj,))_: Keys to be used for the jump trajectories.
        exp_ops _(list of array-like)_: Operators to calculate expectation values of,
            in case some of the cost functions depend on the value of certain
            expectation values.

    Returns:
        _(MCSolveModel)_: Model that when called with the parameters we optimize
            over as argument returns the results of `mcsolve` as well as the updated
            Hamiltonian

    Examples:
        To instantiate a MCSolveModel, we need to pass in state kets for the initial
        states and keys to be used for the jump trajectories. Continuing the `mesolve_model`
        example:
        ```python
        keys = jax.random.split(jax.random.key(31), num=10)
        mc_Kerr_model = ql.mcsolve_model(
            update_H_topt,
            jump_ops,
            initial_states,
            update_tsave_topt,
            keys=keys,
            exp_ops=exp_ops,
        )
        ```
        # TODO add mcsolve tutorial
    """  # noqa E501
    H_function, psi0, tsave_function, exp_ops = _initialize_model(
        H_function, psi0, tsave_or_function, exp_ops
    )
    jump_ops = [_astimearray(L) for L in jump_ops]
    return MCSolveModel(H_function, psi0, tsave_function, exp_ops, jump_ops, keys)


def _initialize_model(
    H_function: callable,
    psi0: ArrayLike,
    tsave_or_function: ArrayLike | callable,
    exp_ops: list[ArrayLike] | None,
) -> [callable, Array, callable, Array]:
    H_function = jtu.Partial(H_function)
    psi0 = jnp.asarray(psi0, dtype=cdtype())
    if callable(tsave_or_function):
        tsave_function = jtu.Partial(tsave_or_function)
    else:
        tsave_function = jtu.Partial(lambda _: tsave_or_function)
    exp_ops = jnp.asarray(exp_ops, dtype=cdtype()) if exp_ops is not None else None
    return H_function, psi0, tsave_function, exp_ops


class Model(eqx.Module):
    H_function: callable
    initial_states: Array
    tsave_function: callable
    exp_ops: Array | None

    def __call__(
        self,
        parameters: Array | dict,
        solver: Solver = Tsit5(),  # noqa B008
        root_finder: optx.AbstractRootFinder = None,
        gradient: Gradient | None = None,
        options: OptimizerOptions = OptimizerOptions(),  # noqa B008
    ) -> tuple[Result, TimeArray]:
        raise NotImplementedError


class SESolveModel(Model):
    r"""Model for Schrödinger-equation optimization.

    When called with the parameters we optimize over returns the results of `sesolve`
    as well as the updated Hamiltonian.
    """

    def __call__(
        self,
        parameters: Array | dict,
        solver: Solver = Tsit5(),  # noqa B008
        root_finder: optx.AbstractRootFinder = None,  # noqa ARG002
        gradient: Gradient | None = None,
        options: OptimizerOptions = OptimizerOptions(),  # noqa B008
    ) -> tuple[Result, TimeArray]:
        new_H = self.H_function(parameters)
        new_tsave = self.tsave_function(parameters)
        result = dq.sesolve(
            new_H,
            self.initial_states,
            new_tsave,
            exp_ops=self.exp_ops,
            solver=solver,
            gradient=gradient,
            options=options,
        )
        return result, new_H


class MESolveModel(Model):
    r"""Model for Lindblad-master-equation optimization.

    When called with the parameters we optimize over returns the results of `mesolve`
    as well as the updated Hamiltonian.
    """

    jump_ops: list[TimeArray]

    def __call__(
        self,
        parameters: Array | dict,
        solver: Solver = Tsit5(),  # noqa B008
        root_finder: optx.AbstractRootFinder = None,  # noqa ARG002
        gradient: Gradient | None = None,
        options: OptimizerOptions = OptimizerOptions(),  # noqa B008
    ) -> tuple[Result, TimeArray]:
        new_H = self.H_function(parameters)
        new_tsave = self.tsave_function(parameters)
        result = dq.mesolve(
            new_H,
            self.jump_ops,
            self.initial_states,
            new_tsave,
            exp_ops=self.exp_ops,
            solver=solver,
            gradient=gradient,
            options=options,
        )
        return result, new_H


class MCSolveModel(Model):
    r"""Model for Monte-Carlo wave function optimization.

    When called with the parameters we optimize over returns the results of `mcsolve`
    as well as the updated Hamiltonian.
    """

    jump_ops: list[TimeArray]
    keys: Array

    def __call__(
        self,
        parameters: Array | dict,
        solver: Solver = Tsit5(),  # noqa B008
        root_finder: optx.AbstractRootFinder = optx.Newton(1e-5, 1e-5, optx.rms_norm),  # noqa: B008
        gradient: Gradient | None = None,
        options: OptimizerOptions = OptimizerOptions(),  # noqa B008
    ) -> tuple[Result, TimeArray]:
        new_H = self.H_function(parameters)
        new_tsave = self.tsave_function(parameters)
        result = dq.mcsolve(
            new_H,
            self.jump_ops,
            self.initial_states,
            new_tsave,
            keys=self.keys,
            exp_ops=self.exp_ops,
            solver=solver,
            root_finder=root_finder,
            gradient=gradient,
            options=options,
        )
        return result, new_H
