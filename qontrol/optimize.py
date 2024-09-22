from __future__ import annotations

import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from dynamiqs.gradient import Gradient
from dynamiqs.solver import Solver, Tsit5
from jax import Array
from jaxtyping import ArrayLike
from optax import GradientTransformation, OptState, TransformInitFn

from .cost import Cost, SummedCost
from .model import Model
from .options import OptimizerOptions
from .plot import _plot_controls_and_loss
from .utils.file_io import save_optimization

TERMINATION_MESSAGES = {
    -1: 'terminated on keyboard interrupt',
    0: 'reached maximum number of allowed epochs;',
    1: '`gtol` termination condition is satisfied;',
    2: '`ftol` termination condition is satisfied;',
    3: '`xtol` termination condition is satisfied;',
    4: 'target cost reached for one cost function;',
    5: 'target cost reached for all cost functions;',
}


def optimize(
    parameters: ArrayLike | dict,
    costs: Cost,
    model: Model,
    *,
    optimizer: GradientTransformation = optax.adam(0.0001, b1=0.99, b2=0.99),  # noqa: B008
    solver: Solver = Tsit5(),  # noqa: B008
    gradient: Gradient | None = None,
    options: OptimizerOptions = OptimizerOptions(),  # noqa: B008
    filepath: str | None = None,
) -> Array | dict:
    r"""Perform gradient descent to optimize Hamiltonian parameters.

    This function takes as input `parameters` which parametrize a `model` when called
    performs time-dynamics simulations using dynamiqs. How to update `parameters` is encoded
    in the list of cost functions `costs` that contains e.g. infidelity contributions, pulse
    amplitude penalties, etc.

    Args:
        parameters _(dict or array-like)_: parameters to optimize
            over that are used to define the Hamiltonian and control times.
        costs _(list of Cost instances)_: List of cost functions used to perform the
            optimization.
        model _(Model)_: Model that is called at each iteration step.
        optimizer _(optax.GradientTransformation)_: optax optimizer to use
            for gradient descent. Defaults to the Adam optimizer.
        solver _(Solver)_: Solver passed to dynamiqs.
        gradient _()Gradient_: Gradient passed to dynamiqs.
        options _(OptimizerOptions)_: Options for grape optimization and dynamiqs
            integration.
        filepath _(str)_: Filepath of where to save optimization results.

    Returns:
        optimized parameters from the final timestep
    """  # noqa E501
    # initialize
    opt_state = optimizer.init(parameters)
    cost_values_over_epochs = []
    epoch_times = []
    previous_parameters = parameters
    prev_total_cost = 0.0

    @partial(jax.jit, static_argnames=('_solver', '_gradient', '_options'))
    def step(
        _parameters: ArrayLike | dict,
        _costs: Cost,
        _model: Model,
        _opt_state: OptState,
        _solver: Solver,
        _gradient: Gradient,
        _options: OptimizerOptions,
    ) -> [Array, TransformInitFn, Array]:
        grads, aux = jax.grad(loss, has_aux=True)(
            _parameters, _costs, _model, _solver, _gradient, _options
        )
        updates, _opt_state = optimizer.update(grads, _opt_state)
        _parameters = optax.apply_updates(_parameters, updates)
        return _parameters, grads, _opt_state, aux

    if options.verbose and filepath is not None:
        print(f'saving results to {filepath}')
    try:  # trick for catching keyboard interrupt
        for epoch in range(options.epochs):
            epoch_start_time = time.time()
            parameters, grads, opt_state, aux = step(
                parameters, costs, model, opt_state, solver, gradient, options
            )
            elapsed_time = np.around(time.time() - epoch_start_time, decimals=3)
            total_cost, cost_values, terminate_for_cost, expects = aux
            cost_values_over_epochs.append(cost_values)
            epoch_times.append(elapsed_time)
            if options.verbose:
                print(f'epoch: {epoch}, elapsed_time: {elapsed_time} s; ')
                if isinstance(costs, SummedCost):
                    for _cost, _cost_value in zip(costs.costs, cost_values):
                        print(_cost, ' = ', _cost_value, '; ', end=' ')
                    print('\n')
                else:
                    print(costs, cost_values[0])
            if filepath is not None:
                save_optimization(
                    filepath,
                    {'cost_values': jnp.asarray(cost_values)},
                    parameters,
                    options.__dict__,
                    epoch,
                )
            if options.plot and epoch % options.plot_period == 0:
                _plot_controls_and_loss(
                    parameters,
                    costs,
                    model,
                    expects,
                    cost_values_over_epochs,
                    epoch,
                    options,
                )
            # early termination
            termination_key = _terminate_early(
                grads,
                parameters,
                previous_parameters,
                total_cost,
                prev_total_cost,
                terminate_for_cost,
                epoch,
                options,
            )
            if termination_key != -1:
                break
            previous_parameters = parameters
            prev_total_cost = total_cost
    except KeyboardInterrupt:
        pass
    if options.plot:
        _plot_controls_and_loss(
            parameters,
            costs,
            model,
            expects,
            cost_values_over_epochs,
            len(cost_values_over_epochs) - 1,
            options,
        )
    print(TERMINATION_MESSAGES[termination_key])
    print(
        f'optimization terminated after {epoch} epochs; \n'
        f'average epoch time (excluding jit) of '
        f'{np.around(np.mean(epoch_times[1:]), decimals=5)} s; \n'
        f'max epoch time of {np.max(epoch_times[1:])} s; \n'
        f'min epoch time of {np.min(epoch_times[1:])} s'
    )
    return parameters


def loss(
    parameters: Array | dict,
    costs: Cost,
    model: Model,
    solver: Solver,
    gradient: Gradient,
    options: OptimizerOptions,
) -> [float, Array]:
    result, H = model(parameters, solver, gradient, options)
    cost_values, terminate = zip(*costs(result, H))
    total_cost = jax.tree.reduce(jnp.add, cost_values)
    total_cost = jnp.log(jnp.sum(jnp.asarray(total_cost)))
    return total_cost, (total_cost, cost_values, terminate, result.expects)


def _terminate_early(
    grads: Array | dict,
    parameters: Array | dict,
    previous_parameters: Array | dict,
    total_cost: Array,
    prev_total_cost: Array,
    terminate_for_cost: list[bool],
    epoch: int,
    options: OptimizerOptions,
) -> None | int:
    termination_key = -1
    if epoch == options.epochs - 1:
        termination_key = 0
    # gtol and xtol
    dx = 0.0
    dg = 0.0
    if isinstance(parameters, dict):
        for (_key_new, val_new), (_key_old, val_old) in zip(
            parameters.items(), previous_parameters.items()
        ):
            dx += _norm(val_new - val_old)
        for _, grad_val in grads.items():
            dg += _norm(grad_val)
    else:
        dx += _norm(parameters - previous_parameters)
        dg += _norm(grads)
    if dg < options.gtol:
        termination_key = 1
    # ftol
    dF = np.abs(total_cost - prev_total_cost)
    if dF < options.ftol * total_cost:
        termination_key = 2
    if dx < options.xtol * (options.xtol + dx):
        termination_key = 3
    if not options.all_costs and any(terminate_for_cost):
        termination_key = 4
    if options.all_costs and all(terminate_for_cost):
        termination_key = 5

    return termination_key


def _norm(x: Array) -> Array:
    if x.shape == ():
        return np.abs(x)
    return np.linalg.norm(x, ord=np.inf)
