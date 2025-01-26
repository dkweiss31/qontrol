from __future__ import annotations

import dynamiqs as dq
import equinox as eqx
import jax.tree_util as jtu
from dynamiqs import QArrayLike, TimeQArray
from dynamiqs.gradient import Gradient
from dynamiqs.result import Result
from dynamiqs.solver import Solver, Tsit5
from jax import Array
from jaxtyping import ArrayLike


def sesolve_model(
    H_function: callable,
    psi0: QArrayLike,
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
        psi0 _(QArrayLike of shape (..., n, 1))_: Initial states.
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
    H_function, tsave_or_function = _initialize_model(H_function, tsave_or_function)
    return SESolveModel(H_function, psi0, tsave_or_function, exp_ops=exp_ops)


def mesolve_model(
    H_function: callable,
    jump_ops: list[QArrayLike | TimeQArray],
    rho0: QArrayLike,
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
        jump_ops _(list of qarray-like or time-qarray, each of shape (...Lk, n, n))_:
            List of jump operators.
        rho0 _(QArrayLike of shape (..., n, n))_: Initial density matrices.
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
    H_function, tsave_function = _initialize_model(H_function, tsave_or_function)
    return MESolveModel(
        H_function, rho0, tsave_function, exp_ops=exp_ops, jump_ops=jump_ops
    )


def sepropagator_model(
    H_function: callable,
    tsave_or_function: ArrayLike | callable,
    *,
    exp_ops: list[ArrayLike] | None = None,
) -> SEPropagatorModel:
    r"""Instantiate sepropagator model.

    Here we instantiate the model that is called at each step of the optimization
    iteration, returning a tuple of the result of calling `sepropagator` as well as the
    Hamiltonian evaluated at the parameter values.

    Args:
        H_function _(callable)_: function specifying how to update the Hamiltonian
        tsave_or_function _(ArrayLike of shape (ntsave,) or callable)_: Either an
            array of times passed to sesolve or a method specifying how to update
            the times that are passed to sesolve
        exp_ops _(list of array-like)_: Operators to calculate expectation values of,
            in case some of the cost functions depend on the value of certain
            expectation values.

    Returns:
        _(SEPropagateModel)_: Model that when called with the parameters we optimize
            over as argument returns the results of `aepropagator` as well as the updated
            Hamiltonian

    Examples:
        

    """
    H_function, tsave_or_function = _initialize_model(H_function, tsave_or_function)
    return SEPropagatorModel(H_function, tsave_or_function, exp_ops=exp_ops)


def mepropagator_model(
    H_function: callable,
    tsave_or_function: ArrayLike | callable,
    *,
    exp_ops: list[ArrayLike] | None = None,
) -> MEPropagatorModel:
    r"""Instantiate mepropagator model.

    Here we instantiate the model that is called at each step of the optimization
    iteration, returning a tuple of the result of calling `mepropagator` as well as the
    Hamiltonian evaluated at the parameter values.

    Args:
        H_function _(callable)_: function specifying how to update the Hamiltonian
        tsave_or_function _(ArrayLike of shape (ntsave,) or callable)_: Either an
            array of times passed to sesolve or a method specifying how to update
            the times that are passed to sesolve
        exp_ops _(list of array-like)_: Operators to calculate expectation values of,
            in case some of the cost functions depend on the value of certain
            expectation values.

    Returns:
        _(MEPropagateModel)_: Model that when called with the parameters we optimize
            over as argument returns the results of `mepropagator` as well as the updated
            Hamiltonian

    Examples:
        

    """  # noqa E501
    H_function, tsave_or_function = _initialize_model(H_function, tsave_or_function)
    return MEPropagatorModel(H_function, tsave_or_function, exp_ops=exp_ops)


def _initialize_model(
    H_function: callable, tsave_or_function: ArrayLike | callable
) -> [callable, callable]:
    H_function = jtu.Partial(H_function)
    if callable(tsave_or_function):
        tsave_function = jtu.Partial(tsave_or_function)
    else:
        tsave_function = jtu.Partial(lambda _: tsave_or_function)
    return H_function, tsave_function


class Model(eqx.Module):
    H_function: callable
    initial_states: QArrayLike
    tsave_function: callable
    exp_ops: Array | None

    def __call__(
        self,
        parameters: Array | dict,
        solver: Solver = Tsit5(),  # noqa B008
        gradient: Gradient | None = None,
        options: dq.Options = dq.Options(),  # noqa B008
    ) -> tuple[Result, TimeQArray]:
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
        gradient: Gradient | None = None,
        options: dq.Options = dq.Options(),  # noqa B008
    ) -> tuple[Result, TimeQArray]:
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

    jump_ops: list[QArrayLike | TimeQArray]

    def __call__(
        self,
        parameters: Array | dict,
        solver: Solver = Tsit5(),  # noqa B008
        gradient: Gradient | None = None,
        options: dq.Options = dq.Options(),  # noqa B008
    ) -> tuple[Result, TimeQArray]:
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


class SEPropagatorModel(Model):
    r"""Model for Schrödinger-equation propagator optimization.

    When called with the parameters we optimize over returns the results of `sepropagate`
    as well as the updated Hamiltonian.
    """

    def __call__(
        self,
        parameters: Array | dict,
        solver: Solver = Tsit5(),  # noqa B008
        gradient: Gradient | None = None,
        options: dq.Options = dq.Options(),  # noqa B008
    ) -> tuple[Result, TimeQArray]:
        new_H = self.H_function(parameters)
        new_tsave = self.tsave_function(parameters)
        result = dq.sepropagator(
            new_H,
            new_tsave,
            exp_ops=self.exp_ops,
            solver=solver,
            gradient=gradient,
            options=options,
        )
        return result, new_H


class MEPropagatorModel(Model):
    r"""Model for Lindblad-master-equation propagator optimization.

    When called with the parameters we optimize over returns the results of `mesolve`
    as well as the updated Hamiltonian.
    """

    jump_ops: list[QArrayLike | TimeQArray]

    def __call__(
        self,
        parameters: Array | dict,
        solver: Solver = Tsit5(),  # noqa B008
        gradient: Gradient | None = None,
        options: dq.Options = dq.Options(),  # noqa B008
    ) -> tuple[Result, TimeQArray]:
        new_H = self.H_function(parameters)
        new_tsave = self.tsave_function(parameters)
        result = dq.mepropagator(
            new_H,
            self.jump_ops,
            new_tsave,
            exp_ops=self.exp_ops,
            solver=solver,
            gradient=gradient,
            options=options,
        )
        return result, new_H
