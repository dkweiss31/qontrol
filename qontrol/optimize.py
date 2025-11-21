from __future__ import annotations

import time
import warnings
from collections.abc import Callable

import dynamiqs as dq
import jax
import jax.numpy as jnp
import numpy as np
import optax
from dynamiqs.gradient import Gradient
from dynamiqs.method import Method, Tsit5
from jax import Array
from jaxtyping import ArrayLike
from optax import GradientTransformation, OptState, TransformInitFn

from .cost import Cost, SummedCost
from .model import Model, SolveModel
from .plot import DefaultPlotter, plot_controls, plot_fft, Plotter
from .recorder import OptimizerRecorder
from .utils.file_io import append_to_h5


TERMINATION_MESSAGES = {
    -1: 'terminated on keyboard interrupt',
    0: 'reached maximum number of allowed epochs',
    1: '`gtol` termination condition is satisfied',
    2: '`xtol` termination condition is satisfied',
    3: '`ftol` termination condition is satisfied',
    4: 'target cost reached for all cost functions',
}

default_options = {
    'verbose': True,
    'epochs': 2000,
    'batch_initial_parameters': False,
    'plot': True,
    'plot_period': 30,
    'save_period': 30,
    'xtol': 1e-8,
    'ftol': 1e-8,
    'gtol': 1e-8,
}


def optimize(
    parameters: ArrayLike | dict,
    costs: Cost,
    model: Model,
    *,
    optimizer: GradientTransformation = optax.adam(0.0001, b1=0.99, b2=0.99),  # noqa: B008
    plotter: Plotter | None = None,
    method: Method = Tsit5(),  # noqa: B008
    gradient: Gradient | None = None,
    dq_options: dq.Options = dq.Options(),  # noqa: B008
    opt_options: dict | None = None,
    filepath: str | None = None,
) -> Array | dict:
    r"""Perform gradient descent to optimize Hamiltonian parameters.

    This function takes as input `parameters` which parametrize a `model` when called
    performs time-dynamics simulations using Dynamiqs. How to update `parameters` is
    encoded in the list of cost functions `costs` that contains e.g. infidelity
    contributions, pulse amplitude penalties, etc.

    Parameters:
        parameters: parameters to optimize over that are used to define the Hamiltonian
            and control times.
        costs: List of cost functions used to perform the optimization.
        model: Model that is called at each iteration step.
        optimizer: optax optimizer to use
            for gradient descent. Defaults to the Adam optimizer.
        plotter: Plotter for monitoring the optimization.
        method: Method passed to Dynamiqs.
        gradient: Gradient passed to Dynamiqs.
        filepath: Filepath of where to save optimization results.
        dq_options : Options for the Dynamiqs integrator.
        opt_options: Options for grape optimization.
            ??? info "Detailed `opt_options` API"
                - `verbose` (`bool`, default: `True`): If `True`, the optimizer will
                    print out the infidelity at each epoch step to track the progress of
                    the optimization.
                - `epochs` (`int`, default: `2000`): Number of optimization epochs.
                - `batch_initial_parameters` (`bool`, default: False): Whether to batch
                    over initial parameters. If True, then `len(parameters)` defines the
                    number of simulations to batch over. If False, then `parameters` is
                    assumed to not be batched.
                - `plot` (`bool`, default: `True`): Whether to plot the results during
                    the optimization (for the epochs where results are plotted,
                    necessarily suffer a time penalty).
                - `plot_period` (`int`, default: `30`): If plot is `True`, plot every
                    `plot_period`.
                - `save_period` (`int`, default: `30`): If a filepath is provided, save
                    every `save_period`.
                - `xtol` (`float`, default: `1e-8`): Terminate the optimization if the
                    parameters are not being updated.
                - `ftol` (`float`, default: `1e-8`): Terminate the optimization if the
                    cost function is not changing above this level.
                - `gtol` (`float`, default: `1e-8`): Terminate the optimization if the
                    norm of the gradient falls below this level.

    Returns:
        Optimized parameters from the final timestep.
    """
    # check opt_option keys and deprecated options
    for key in opt_options:
        if key not in default_options:
            raise ValueError(f'{key} not a valid option')
    if 'ignore_termination' in opt_options:
        warnings.warn(
            "'ignore_termination' no longer accepted as an option and is now ignored",
            DeprecationWarning,
            stacklevel=2,
        )
        opt_options.pop('ignore_termination')
    if 'all_costs' in opt_options:
        warnings.warn(
            "'all_costs' no longer accepted as an option and is now ignored: all cost"
            ' functions must be below their target for the optimization to terminate.',
            DeprecationWarning,
            stacklevel=2,
        )
        opt_options.pop('all_costs')
    # initialize with default options for those that aren't specified
    opt_options = {**default_options, **(opt_options or {})}

    if (
        opt_options['batch_initial_parameters']
        and isinstance(parameters, list)
        and isinstance(parameters[0], dict)
    ):
        raise ValueError('batching with lists of dicts not supported')

    opt_recorder = OptimizerRecorder(parameters)
    if filepath is not None:
        print(f'saving results to {filepath}')

    plotter = _initialize_plotter(plotter, model, opt_options)
    step_fn, opt_state = _setup_optimization(
        parameters, costs, model, optimizer, method, gradient, dq_options, opt_options
    )
    epoch = 0
    termination_key = -1
    # trick for catching keyboard interrupt
    try:
        for epoch in range(opt_options['epochs']):
            parameters, grads, opt_state, aux = _run_epoch(
                parameters,
                costs,
                model,
                plotter,
                opt_options,
                filepath,
                epoch,
                opt_recorder,
                opt_state,
                step_fn,
            )

            terminate, termination_key = _check_for_termination(
                opt_options, aux, epoch, grads, opt_recorder
            )
            if terminate:
                break

            opt_recorder.previous_parameters = parameters

    except KeyboardInterrupt:
        pass

    if epoch > 0:
        # save any unsaved data and make a final plot
        total_cost, _, _, expects = aux
        if filepath is not None:
            append_to_h5(filepath, opt_recorder.data_to_save(), opt_options)
        carry = total_cost, opt_recorder.cost_values, expects, epoch, True
        _plot(parameters, costs, model, plotter, opt_options, carry)
        times = opt_recorder.epoch_times[1:]
        print(
            f'{TERMINATION_MESSAGES[termination_key]} \n'
            f'optimization terminated after {epoch} epochs \n'
            f'average epoch time (excluding jit) of {np.mean(times):.5f} s \n'
            f'max epoch time of {np.max(times):.5f} s \n'
            f'min epoch time of {np.min(times):.5f} s'
        )
        if filepath is not None:
            print(f'results saved to {filepath}')

    return parameters


def loss(
    parameters: Array | dict,
    costs: Cost,
    model: Model,
    method: Method,
    gradient: Gradient,
    dq_options: dq.Options,
) -> tuple[Array, tuple[Array, Array, Array, Array]]:
    result, H = model(parameters, method, gradient, dq_options)
    cost_values, terminate = zip(*costs(result, H, parameters), strict=True)
    total_cost = jnp.log(jax.tree.reduce(jnp.add, cost_values))
    expects = result.expects if hasattr(result, 'expects') else None
    return total_cost, (
        total_cost,
        jnp.asarray(cost_values),
        jnp.asarray(terminate),
        expects,
    )


def _plot(
    parameters: Array | dict,
    costs: Cost,
    model: Model,
    plotter: Plotter,
    opt_options: dict,
    carry: tuple,
):
    total_cost, cost_values_over_epochs, expects, epoch, override = carry
    if opt_options['plot'] and (
        override or (epoch % opt_options['plot_period'] == 0 and epoch > 0)
    ):
        if opt_options['batch_initial_parameters']:
            # plot for the lowest cost
            minimum_cost_idx = np.argmin(total_cost)
            _cost_values_over_epochs = np.array(cost_values_over_epochs)[
                :, minimum_cost_idx
            ]
            _expects = expects[minimum_cost_idx] if expects is not None else None
            plotter.update_plots(
                parameters[minimum_cost_idx],
                costs,
                model,
                _expects,
                _cost_values_over_epochs,
                epoch,
            )
        else:
            plotter.update_plots(
                parameters, costs, model, expects, cost_values_over_epochs, epoch
            )


def _run_epoch(
    parameters: Array | dict,
    costs: Cost,
    model: Model,
    plotter: Plotter,
    opt_options: dict,
    filepath: str,
    epoch: int,
    opt_recorder: OptimizerRecorder,
    opt_state: OptState,
    step_fn: Callable,
) -> tuple[Array | dict, TransformInitFn, OptState, tuple]:
    start_time = time.time()
    parameters, grads, opt_state, aux = jax.block_until_ready(
        step_fn(parameters, opt_state)
    )
    elapsed = time.time() - start_time

    total_cost, cost_values, _, expects = aux
    opt_recorder.record_epoch(parameters, cost_values, elapsed, total_cost)

    if opt_options['verbose']:
        print(f'epoch: {epoch}, elapsed_time: {elapsed:.6f} s; ')
        if isinstance(costs, SummedCost):
            for cost, value in zip(costs.costs, cost_values.T, strict=True):
                print(cost, ' = ', value, '; ', end=' ')
            print('\n')
        else:
            print(costs, np.squeeze(cost_values))

    if filepath is not None and epoch > 0 and epoch % opt_options['save_period'] == 0:
        append_to_h5(filepath, opt_recorder.data_to_save(), opt_options)
        opt_recorder.reset(epoch)
    carry = total_cost, opt_recorder.cost_values, expects, epoch, False
    _plot(parameters, costs, model, plotter, opt_options, carry)
    return parameters, grads, opt_state, aux


def _initialize_plotter(
    plotter: Plotter | None, model: Model, opt_options: dict
) -> Plotter | None:
    # Either a plotter has been provided, or we don't want to plot, in which case we can
    # just return `plotter`
    if isinstance(plotter, Plotter) or not opt_options['plot']:
        return plotter
    # This one includes plotting expectation values, hence the check
    if (
        isinstance(model, SolveModel)
        and model.exp_ops is not None
        and len(model.exp_ops) > 0
    ):
        return DefaultPlotter()
    # If we want to plot, haven't provided a plotter and don't plot expectation values,
    # then we end up here
    return Plotter([plot_fft, plot_controls])


def _setup_optimization(
    parameters: Array | dict,
    costs: Cost,
    model: Model,
    optimizer: GradientTransformation,
    method: Method,
    gradient: Gradient,
    dq_options: dq.Options,
    opt_options: dict,
) -> tuple[Callable, OptState]:
    """Setup Jit-compiled step function and optimizer state."""

    @jax.jit
    def single_step(
        _parameters: ArrayLike | dict, _opt_state: OptState
    ) -> tuple[Array, TransformInitFn, OptState, tuple]:
        grads, aux = jax.grad(loss, has_aux=True)(
            _parameters, costs, model, method, gradient, dq_options
        )
        updates, _opt_state = optimizer.update(grads, _opt_state)
        _parameters = optax.apply_updates(_parameters, updates)
        return _parameters, grads, _opt_state, aux

    @jax.jit
    def batch_step(
        _parameters: ArrayLike | dict, _opt_state: OptState
    ) -> tuple[Array, TransformInitFn, OptState, tuple]:
        return jax.vmap(single_step, in_axes=(0, 0))(_parameters, _opt_state)

    if opt_options['batch_initial_parameters']:
        opt_state = [optimizer.init(param) for param in parameters]
        opt_state = jax.tree.map(lambda *args: jnp.stack(args), *opt_state)
        return batch_step, opt_state
    return single_step, optimizer.init(parameters)


def _check_for_termination(  # noqa PLR0911
    opt_options: dict,
    aux: tuple,
    epoch: int,
    grads: Array | dict,
    opt_recorder: OptimizerRecorder,
) -> tuple[bool, int]:
    _, _, terminate_for_cost, _ = aux
    # Don't do anything if we're in the first epoch (nothing to compare to)
    if epoch == 0:
        return False, -1
    if epoch == opt_options['epochs'] - 1:
        return True, 0
    # Calculate parameter and gradient norms
    dx = _calculate_parameter_diff(opt_recorder)
    dg = _calculate_total_norm(grads)
    if dg < opt_options['gtol']:
        return True, 1
    if dx < opt_options['xtol'] * (opt_options['xtol'] + dx):
        return True, 2
    # Check df (if cost meaningfully changed)
    if _check_cost_tolerance(opt_recorder, opt_options):
        return True, 3
    # Check if all costs below targets
    if _check_cost_targets(terminate_for_cost, opt_options['batch_initial_parameters']):
        return True, 4
    return False, -1


def _calculate_parameter_diff(opt_recorder: OptimizerRecorder) -> np.ndarray | float:
    current = opt_recorder.current_parameters
    previous = opt_recorder.previous_parameters
    if isinstance(current, dict):
        return sum(
            _norm(curr_val - prev_val)
            for (_, curr_val), (_, prev_val) in zip(
                current.items(), previous.items(), strict=True
            )
        )
    return _norm(current - previous)


def _calculate_total_norm(values: Array | dict) -> np.ndarray | float:
    if isinstance(values, dict):
        return sum(_norm(val) for val in values.values())
    return _norm(values)


def _check_cost_tolerance(opt_recorder: OptimizerRecorder, opt_options: dict) -> bool:
    """Check if cost change is below tolerance."""
    current_total_cost, prev_total_cost = opt_recorder.total_costs[-2:]
    cost_diff = np.abs(current_total_cost - prev_total_cost)
    if opt_options['batch_initial_parameters']:
        max_idx = np.argmax(cost_diff)
        return cost_diff[max_idx] < opt_options['ftol'] * current_total_cost[max_idx]
    return cost_diff < opt_options['ftol'] * current_total_cost


def _check_cost_targets(terminate_for_cost: Array, is_batch: bool) -> bool:
    """Check if cost targets have been met.

    For batching, if at least one sim has all cost targets met, then break.
    """
    if is_batch:
        return any(np.all(terminate_for_cost, axis=1))
    return all(terminate_for_cost)


def _norm(x: Array) -> np.ndarray | float:
    if x.shape == ():
        return np.abs(x)
    return np.linalg.norm(x)
