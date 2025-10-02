from __future__ import annotations

import dynamiqs as dq
import equinox as eqx
import jax.tree_util as jtu
from dynamiqs import QArrayLike, TimeQArray
from dynamiqs.gradient import Gradient
from dynamiqs.method import Method, Tsit5
from dynamiqs.result import Result
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

    Here we instantiate the model that is called at each step of the optimization iteration, 
    returning a tuple of the result of calling `sesolve` as well as the Hamiltonian evaluated at 
    the parameter values.

    Parameters:
        H_function: function specifying how to update the Hamiltonian
        psi0: _Shape = (..., n, 1)_. Initial states.
        tsave_or_function: _If ArrayLike, then Shape = (ntsave,)_. Either an array of times passed 
            to `sesolve` or a method specifying how to update the times that are passed to 
            `sesolve`.
        exp_ops: Operators to calculate expectation values of, in case some of the cost functions 
            depend on the value of certain expectation values.

    Returns:
        Model that when called with the parameters we optimize over as argument returns the results 
            of `sesolve` as well as the updated Hamiltonian

    ??? example "Basic example: Piecewise-constant control"
        ```python
        tsave = jnp.linspace(0.0, 11.0, 10)
        psi0 = [dq.basis(2, 0)]
        H1s = [dq.sigmax(), dq.sigmay()]


        def H_pwc(values: Array) -> dq.TimeQArray:
            H = dq.sigmaz()
            for idx, _H1 in enumerate(H1s):
                H += dq.pwc(tsave, values[idx], _H1)
            return H


        sesolve_model = ql.sesolve_model(H_pwc, psi0, tsave)
        ```
        See for example [this tutorial](../examples/qubit).

    ??? example "Advanced example: Spline control"
        In more complex cases, we can imagine that the optimized parameters
        are the control points fed into a spline, and additionally the control
        times themselves are optimized.
        ```python
        init_drive_params_topt = {
            'dp': -0.001 * jnp.ones((len(H1s), len(tsave))),
            't': tsave[-1],
        }


        def H_func_topt(t: float, drive_params_dict: dict) -> dq.TimeQArray:
            drive_params = drive_params_dict['dp']
            new_tsave = jnp.linspace(0.0, drive_params_dict['t'], len(tsave))
            drive_spline = _drive_spline(drive_params, envelope, new_tsave)
            drive_amps = drive_spline.evaluate(t)
            drive_Hs = jnp.einsum('d,dij->ij', drive_amps, H1s)
            return H0 + drive_Hs


        def update_H_topt(drive_params_dict: dict) -> dq.TimeQArray:
            new_H = jtu.Partial(H_func_topt, drive_params_dict=drive_params_dict)
            return dq.timecallable(new_H)


        def update_tsave_topt(drive_params_dict: dict) -> jax.Array:
            return jnp.linspace(0.0, drive_params_dict['t'], len(tsave))


        se_t_opt_Kerr_model = ql.sesolve_model(update_H_topt, psi0, update_tsave_topt)
        ```
        See for example [this tutorial](../examples/Kerr_oscillator#time-optimal-control).

    """
    H_function, tsave_or_function = _initialize_model(H_function, tsave_or_function)
    return SESolveModel(
        H_function, tsave_or_function, exp_ops=exp_ops, initial_states=psi0
    )


def mesolve_model(
    H_function: callable,
    jump_ops: list[QArrayLike | TimeQArray],
    rho0: QArrayLike,
    tsave_or_function: ArrayLike | callable,
    *,
    exp_ops: list[ArrayLike] | None = None,
) -> MESolveModel:
    r"""Instantiate mesolve model.

    Here we instantiate the model that is called at each step of the optimization iteration, 
    returning a tuple of the result of calling `mesolve` as well as the Hamiltonian evaluated at 
    the parameter values.

    Parameters:
        H_function: function specifying how to update the Hamiltonian
        jump_ops: _Each of QArray or TimeQArray is has shape = (...Lk, n, n))_. List of jump 
            operators.
        rho0: _Shape = (..., n, n)_. Initial density matrices.
        tsave_or_function: _If ArrayLike, then Shape = (ntsave,)_. Either an array of times passed 
            to sesolve or a method specifying how to update the times that are passed to `mesolve`.
        exp_ops: Operators to calculate expectation values of, in case some of the cost functions 
            depend on the value of certain expectation values.

    Returns:
        Model that when called with the parameters we optimize over as argument returns the results 
            of `mesolve` as well as the updated Hamiltonian

    ??? example "Advanced example: Spline control"
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
        H_function,
        tsave_function,
        exp_ops=exp_ops,
        initial_states=rho0,
        jump_ops=jump_ops,
    )


def sepropagator_model(
    H_function: callable, tsave_or_function: ArrayLike | callable
) -> SEPropagatorModel:
    r"""Instantiate sepropagator model.

    Here we instantiate the model that is called at each step of the optimization iteration, 
    returning a tuple of the result of calling `sepropagator` as well as the Hamiltonian evaluated 
    at the parameter values.

    Parameters:
        H_function: function specifying how to update the Hamiltonian
        tsave_or_function: _If ArrayLike, then Shape = (ntsave,)_. Either an array of times passed 
            to `sepropagator` or a method specifying how to update the times that are passed to 
            `sepropagator`.

    Returns:
        Model that when called with the parameters we optimize over as argument returns the results 
            of `sepropagator` as well as the updated Hamiltonian

    ??? example "Basic example: Piecewise-constant control"
        ```python
        tsave = jnp.linspace(0.0, 11.0, 10)
        H1s = [dq.sigmax(), dq.sigmay()]


        def H_pwc(values: Array) -> dq.TimeQArray:
            H = dq.sigmaz()
            for idx, _H1 in enumerate(H1s):
                H += dq.pwc(tsave, values[idx], _H1)
            return H


        sepropagator_model = ql.sepropagator_model(H_pwc, tsave)
        ```

    ??? example "Advanced example: Spline control"
        In more complex cases, we can imagine that the optimized parameters
        are the control points fed into a spline, and additionally the control
        times themselves are optimized.
        ```python
        init_drive_params_topt = {
            'dp': -0.001 * jnp.ones((len(H1s), len(tsave))),
            't': tsave[-1],
        }


        def H_func_topt(t: float, drive_params_dict: dict) -> dq.TimeQArray:
            drive_params = drive_params_dict['dp']
            new_tsave = jnp.linspace(0.0, drive_params_dict['t'], len(tsave))
            drive_spline = _drive_spline(drive_params, envelope, new_tsave)
            drive_amps = drive_spline.evaluate(t)
            drive_Hs = jnp.einsum('d,dij->ij', drive_amps, H1s)
            return H0 + drive_Hs


        def update_H_topt(drive_params_dict: dict) -> dq.TimeQArray:
            new_H = jtu.Partial(H_func_topt, drive_params_dict=drive_params_dict)
            return dq.timecallable(new_H)


        def update_tsave_topt(drive_params_dict: dict) -> jax.Array:
            return jnp.linspace(0.0, drive_params_dict['t'], len(tsave))


        sep_t_opt_Kerr_model = ql.sepropagator_model(update_H_topt, update_tsave_topt)
        ```

    """
    H_function, tsave_or_function = _initialize_model(H_function, tsave_or_function)
    return SEPropagatorModel(H_function, tsave_or_function)


def mepropagator_model(
    H_function: callable,
    jump_ops: list[QArrayLike | TimeQArray],
    tsave_or_function: ArrayLike | callable,
) -> MEPropagatorModel:
    r"""Instantiate mepropagator model.

    Here we instantiate the model that is called at each step of the optimization iteration, 
    returning a tuple of the result of calling `mepropagator` as well as the Hamiltonian evaluated 
    at the parameter values.

    Parameters:
        H_function: function specifying how to update the Hamiltonian
        jump_ops: _Each of QArray or TimeQArray is has shape = (...Lk, n, n))_. List of jump 
            operators.
        tsave_or_function: _If ArrayLike, then Shape = (ntsave,)_. Either an array of times passed 
            to `mepropagator` or a method specifying how to update the times that are passed to 
            `mepropagator`.

    Returns:
        Model that when called with the parameters we optimize over as argument returns the results 
            of `mepropagator` as well as the updated Hamiltonian

    ??? example "Advanced example: Spline control"
        Instantiating a `MEPropagatorModel` is quite similar to instantiating an
        `SEPropagatorModel`, with the difference being that we need to supply jump
        operators. Continuing the last example from `sepropagator_model`
        ```python
        jump_ops = [0.03 * dq.sigmax()]
        mep_Kerr_model = ql.mepropagator_model(
            update_H_topt, jump_ops, update_tsave_topt
        )
        ```

    """
    H_function, tsave_or_function = _initialize_model(H_function, tsave_or_function)
    return MEPropagatorModel(H_function, tsave_or_function, jump_ops=jump_ops)


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
    tsave_function: callable

    def __call__(
        self,
        parameters: Array | dict,
        method: Method = Tsit5(),  # noqa B008
        gradient: Gradient | None = None,
        options: dq.Options = dq.Options(),  # noqa B008
    ) -> tuple[Result, TimeQArray]:
        raise NotImplementedError


class SolveModel(Model):
    exp_ops: Array | None


class PropagatorModel(Model):
    pass


class SESolveModel(SolveModel):
    initial_states: QArrayLike

    def __call__(
        self,
        parameters: Array | dict,
        method: Method = Tsit5(),  # noqa B008
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
            method=method,
            gradient=gradient,
            options=options,
        )
        return result, new_H


class MESolveModel(SolveModel):
    initial_states: QArrayLike
    jump_ops: list[QArrayLike | TimeQArray]

    def __call__(
        self,
        parameters: Array | dict,
        method: Method = Tsit5(),  # noqa B008
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
            method=method,
            gradient=gradient,
            options=options,
        )
        return result, new_H


class SEPropagatorModel(PropagatorModel):
    def __call__(
        self,
        parameters: Array | dict,
        method: Method = Tsit5(),  # noqa B008
        gradient: Gradient | None = None,
        options: dq.Options = dq.Options(),  # noqa B008
    ) -> tuple[Result, TimeQArray]:
        new_H = self.H_function(parameters)
        new_tsave = self.tsave_function(parameters)
        result = dq.sepropagator(
            new_H, new_tsave, method=method, gradient=gradient, options=options
        )
        return result, new_H


class MEPropagatorModel(PropagatorModel):
    jump_ops: list[QArrayLike | TimeQArray]

    def __call__(
        self,
        parameters: Array | dict,
        method: Method = Tsit5(),  # noqa B008
        gradient: Gradient | None = None,
        options: dq.Options = dq.Options(),  # noqa B008
    ) -> tuple[Result, TimeQArray]:
        new_H = self.H_function(parameters)
        new_tsave = self.tsave_function(parameters)
        result = dq.mepropagator(
            new_H,
            self.jump_ops,
            new_tsave,
            method=method,
            gradient=gradient,
            options=options,
        )
        return result, new_H
