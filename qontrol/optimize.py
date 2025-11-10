from __future__ import annotations

import copy
import time
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
from .utils.file_io import append_to_h5


TERMINATION_MESSAGES = {
    -1: 'terminated on keyboard interrupt',
    0: 'reached maximum number of allowed epochs;',
    1: '`gtol` termination condition is satisfied;',
    2: '`xtol` termination condition is satisfied;',
    3: '`ftol` termination condition is satisfied;',
    4: 'target cost reached for all cost functions;',
}

default_options = {
    'verbose': True,
    'epochs': 2000,
    'batch': False,
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
                - `batch` (`bool`, default: False): Whether to batch over random
                    initial parameters. If True, then `len(parameters)` defines the
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
    step_fn, opt_state, plotter, opt_recorder, opt_options = _initialize(
        parameters,
        costs,
        model,
        optimizer,
        plotter,
        method,
        gradient,
        dq_options,
        opt_options,
        filepath,
    )
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

    print(TERMINATION_MESSAGES[termination_key])
    if epoch > 0:
        _finalize(
            parameters,
            costs,
            model,
            plotter,
            opt_options,
            filepath,
            aux,
            epoch,
            opt_recorder,
        )

    return parameters


class OptimizerRecorder:
    """Records optimization across epochs."""

    def __init__(self, parameters: Array | dict):
        self.total_costs = []
        self.cost_values = []
        self.epoch_times = []
        if isinstance(parameters, dict):
            init_saved_parameters = {key: [] for key in parameters}
        else:
            init_saved_parameters = []
        self.init_saved_parameters = init_saved_parameters
        self.parameters_since_last_save = copy.deepcopy(init_saved_parameters)
        self.current_parameters = parameters
        self.previous_parameters = None
        self.last_save_epoch = -1

    def _append_parameters(self, parameters: Array | dict):
        if isinstance(parameters, dict):
            for key, val in parameters.items():
                self.parameters_since_last_save[key].append(val)
        else:
            self.parameters_since_last_save.append(parameters)

    def record_epoch(
        self,
        parameters: Array | dict,
        cost_values: Array,
        elapsed_time: float,
        total_cost: Array,
    ):
        """Record results from an epoch."""
        self.current_parameters = parameters
        self._append_parameters(parameters)
        self.total_costs.append(total_cost)
        self.cost_values.append(cost_values)
        self.epoch_times.append(elapsed_time)

    def reset(self, epoch: int):
        """Reset saved parameters after a save operation."""
        self.parameters_since_last_save = copy.deepcopy(self.init_saved_parameters)
        self.last_save_epoch = epoch

    def data_to_save(self) -> dict:
        # don't want to resave data from the epoch we last saved at, so +1
        data_dict = {
            'cost_values': self.cost_values[self.last_save_epoch + 1 :],
            'total_cost': self.total_costs[self.last_save_epoch + 1 :],
        }
        if isinstance(self.parameters_since_last_save, dict):
            data_dict |= self.parameters_since_last_save
        else:
            data_dict['parameters'] = self.parameters_since_last_save
        return data_dict


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
    return total_cost, (total_cost, cost_values, terminate, expects)


def _plot(
    parameters: Array | dict,
    costs: Cost,
    model: Model,
    plotter: Plotter,
    opt_options: dict,
    carry: tuple,
):
    total_cost, cost_values_over_epochs, expects, epoch, override = carry
    if override or (
        opt_options['plot'] and epoch % opt_options['plot_period'] == 0 and epoch > 0
    ):
        if opt_options['batch']:
            # plot for the lowest cost
            minimum_cost_idx = np.argmin(total_cost)
            _cost_values_over_epochs = np.array(cost_values_over_epochs)[
                :, minimum_cost_idx
            ]
            plotter.update_plots(
                parameters[minimum_cost_idx],
                costs,
                model,
                expects[minimum_cost_idx],
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
            for cost, value in zip(costs.costs, cost_values, strict=True):
                print(cost, ' = ', value, '; ', end=' ')
            print('\n')
        else:
            print(costs, cost_values[0])

    if filepath is not None and epoch > 0 and epoch % opt_options['save_period'] == 0:
        append_to_h5(filepath, opt_recorder.data_to_save(), opt_options)
        opt_recorder.reset(epoch)
    carry = total_cost, opt_recorder.cost_values, expects, epoch, False
    _plot(parameters, costs, model, plotter, opt_options, carry)
    return parameters, grads, opt_state, aux


def _initialize(
    parameters: ArrayLike | dict,
    costs: Cost,
    model: Model,
    optimizer: GradientTransformation,
    plotter: Plotter | None,
    method: Method,
    gradient: Gradient | None,
    dq_options: dq.Options,
    opt_options: dict | None,
    filepath: str | None,
) -> tuple[Callable, OptState, Plotter, OptimizerRecorder, dict]:
    opt_options = {**default_options, **(opt_options or {})}
    if (
        opt_options['batch']
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
    return step_fn, opt_state, plotter, opt_recorder, opt_options


def _finalize(
    parameters: Array | dict,
    costs: Cost,
    model: Model,
    plotter: Plotter | None,
    opt_options: dict,
    filepath: str | None,
    aux: tuple,
    epoch: int,
    opt_recorder: OptimizerRecorder,
):
    # save any unsaved data and make a final plot
    total_cost, _, _, expects = aux
    if filepath is not None and epoch > 0:
        append_to_h5(filepath, opt_recorder.data_to_save(), opt_options)
    carry = total_cost, opt_recorder.cost_values, expects, epoch, True
    _plot(parameters, costs, model, plotter, opt_options, carry)
    times = opt_recorder.epoch_times[1:]
    print(
        f'optimization terminated after {epoch} epochs; \n'
        f'average epoch time (excluding jit) of {np.mean(times):.5f} s; \n'
        f'max epoch time of {np.max(times):.5f} s; \n'
        f'min epoch time of {np.min(times):.5f} s'
    )
    if filepath is not None:
        print(f'results saved to {filepath}')


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

    if opt_options['batch']:
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
    if _check_cost_targets(terminate_for_cost, opt_options['batch']):
        return True, 4
    return False, -1


def _calculate_parameter_diff(opt_recorder: OptimizerRecorder) -> np.ndarray | float:
    """Calculate total norm of difference between current and previous values."""
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
    """Calculate total norm of values."""
    if isinstance(values, dict):
        return sum(_norm(val) for val in values.values())
    return _norm(values)


def _check_cost_tolerance(opt_recorder: OptimizerRecorder, opt_options: dict) -> bool:
    """Check if cost change is below tolerance."""
    current_total_cost, prev_total_cost = opt_recorder.total_costs[-2:]
    cost_diff = np.abs(current_total_cost - prev_total_cost)
    if opt_options['batch']:
        max_idx = np.argmax(cost_diff)
        return cost_diff[max_idx] < opt_options['ftol'] * current_total_cost[max_idx]
    return cost_diff < opt_options['ftol'] * current_total_cost


def _check_cost_targets(terminate_for_cost: list[bool], is_batch: bool) -> bool:
    """Check if cost targets have been met."""
    if is_batch:
        return any(all(term_for_c) for term_for_c in terminate_for_cost)
    return all(terminate_for_cost)


def _norm(x: Array) -> np.ndarray | float:
    if x.shape == ():
        return np.abs(x)
    return np.linalg.norm(x)
