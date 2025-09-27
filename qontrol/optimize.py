from __future__ import annotations

import time
from functools import partial

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
    2: '`ftol` termination condition is satisfied;',
    3: '`xtol` termination condition is satisfied;',
    4: 'target cost reached for one cost function;',
    5: 'target cost reached for all cost functions;',
}

default_options = {
    'verbose': True,
    'ignore_termination': False,
    'all_costs': True,
    'epochs': 2000,
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
        plotter _(Plotter)_: Plotter for monitoring the optimization.
        method _(Method)_: Method passed to dynamiqs.
        gradient _()Gradient_: Gradient passed to dynamiqs.
        options _(dict)_: Options for grape optimization.
            verbose _(bool)_: If `True`, the optimizer will print out the infidelity at
                each epoch step to track the progress of the optimization.
            ignore_termination _(bool)_: Whether to ignore the various termination conditions
            all_costs _(bool)_: Whether or not all costs must be below their targets for
                early termination of the optimizer. If False, the optimization terminates
                if only one cost function is below the target (typically infidelity).
            epochs _(int)_: Number of optimization epochs.
            plot _(bool)_: Whether to plot the results during the optimization (for the
                epochs where results are plotted, necessarily suffer a time penalty).
            plot_period _(int)_: If plot is True, plot every plot_period. Defaults to 30.
            save_period _(int)_: If filepath is provided, save every save_period. Defaults to 30.
            xtol _(float)_: Defaults to 1e-8, terminate the optimization if the parameters
                are not being updated
            ftol _(float)_: Defaults to 1e-8, terminate the optimization if the cost
                function is not changing above this level
            gtol _(float)_: Defaults to 1e-8, terminate the optimization if the norm of the
                gradient falls below this level
        filepath _(str)_: Filepath of where to save optimization results.

    Returns:
        optimized parameters from the final timestep
    """  # noqa E501
    # initialize
    opt_options = {**default_options, **(opt_options or {})}
    opt_state = optimizer.init(parameters)
    total_cost_over_epochs = []
    cost_values_over_epochs = []
    epoch_times = []

    def _init_saved_parameters(_parameters: ArrayLike | dict) -> list | dict:
        if isinstance(_parameters, dict):
            return {_key: [] for _key, _val in _parameters.items()}
        return [_parameters]

    parameters_since_last_save = _init_saved_parameters(parameters)
    previous_parameters = parameters
    last_save_epoch = 0

    # Initialize plotter if needed
    if plotter is None and opt_options['plot']:
        if (
            isinstance(model, SolveModel)
            and model.exp_ops is not None
            and len(model.exp_ops) > 0
        ):
            plotter = DefaultPlotter()
        else:
            plotter = Plotter([plot_fft, plot_controls])

    @partial(jax.jit, static_argnames=('_method', '_gradient', '_options'))
    def step(
        _parameters: ArrayLike | dict,
        _costs: Cost,
        _model: Model,
        _opt_state: OptState,
        _method: Method,
        _gradient: Gradient,
        _options: dq.Options,
    ) -> [Array, TransformInitFn, Array]:
        grads, aux = jax.grad(loss, has_aux=True)(
            _parameters, _costs, _model, _method, _gradient, _options
        )
        updates, _opt_state = optimizer.update(grads, _opt_state)
        _parameters = optax.apply_updates(_parameters, updates)
        return _parameters, grads, _opt_state, aux

    if opt_options['verbose'] and filepath is not None:
        print(f'saving results to {filepath}')
    try:  # trick for catching keyboard interrupt
        for epoch in range(opt_options['epochs']):
            epoch_start_time = time.time()
            parameters, grads, opt_state, aux = jax.block_until_ready(
                step(parameters, costs, model, opt_state, method, gradient, dq_options)
            )
            elapsed_time = np.around(time.time() - epoch_start_time, decimals=3)

            # Unpack and record results
            total_cost, cost_values, terminate_for_cost, expects = aux
            total_cost_over_epochs.append(total_cost)
            cost_values_over_epochs.append(cost_values)
            epoch_times.append(elapsed_time)
            if isinstance(parameters, dict):
                for key, val in parameters.items():
                    parameters_since_last_save[key].append(val)
            else:
                parameters_since_last_save.append(parameters)

            # Print out the costs for this step
            if opt_options['verbose']:
                print(f'epoch: {epoch}, elapsed_time: {elapsed_time} s; ')
                if isinstance(costs, SummedCost):
                    for _cost, _cost_value in zip(
                        costs.costs, cost_values, strict=True
                    ):
                        print(_cost, ' = ', _cost_value, '; ', end=' ')
                    print('\n')
                else:
                    print(costs, cost_values[0])

            # Save logic
            save_period = opt_options['save_period']
            if filepath is not None and epoch > 0 and epoch % save_period == 0:
                _save(
                    cost_values_over_epochs,
                    total_cost_over_epochs,
                    parameters_since_last_save,
                    last_save_epoch,
                    opt_options,
                    filepath,
                )
                last_save_epoch = epoch
                parameters_since_last_save = _init_saved_parameters(parameters)
            # Plot the cost values as well as other desired quantities
            if (
                opt_options['plot']
                and epoch % opt_options['plot_period'] == 0
                and epoch > 0
            ):
                plotter.update_plots(
                    parameters, costs, model, expects, cost_values_over_epochs, epoch
                )

            # Check for early termination
            if epoch > 0 and not opt_options['ignore_termination']:
                termination_key = _terminate_early(
                    grads,
                    parameters,
                    previous_parameters,
                    total_cost,
                    total_cost_over_epochs[-2],
                    terminate_for_cost,
                    epoch,
                    opt_options,
                )
                if termination_key != -1:
                    break

    except KeyboardInterrupt:
        pass

    # save remaining unsaved data
    if filepath is not None:
        _save(
            cost_values_over_epochs,
            total_cost_over_epochs,
            parameters_since_last_save,
            last_save_epoch,
            opt_options,
            filepath,
        )

    # Final plot update
    if opt_options['plot']:
        plotter.update_plots(
            parameters,
            costs,
            model,
            expects,
            cost_values_over_epochs,
            len(cost_values_over_epochs) - 1,
        )
    if not opt_options['ignore_termination']:
        print(TERMINATION_MESSAGES[termination_key])
    print(
        f'optimization terminated after {epoch} epochs; \n'
        f'average epoch time (excluding jit) of '
        f'{np.around(np.mean(epoch_times[1:]), decimals=5)} s; \n'
        f'max epoch time of {np.max(epoch_times[1:])} s; \n'
        f'min epoch time of {np.min(epoch_times[1:])} s'
    )
    if opt_options['verbose'] and filepath is not None:
        print(f'results saved to {filepath}')
    return parameters


def loss(
    parameters: Array | dict,
    costs: Cost,
    model: Model,
    method: Method,
    gradient: Gradient,
    dq_options: dq.Options,
) -> [float, Array]:
    result, H = model(parameters, method, gradient, dq_options)
    cost_values, terminate = zip(*costs(result, H, parameters), strict=True)
    total_cost = jax.tree.reduce(jnp.add, cost_values)
    total_cost = jnp.log(jnp.sum(jnp.asarray(total_cost)))
    expects = result.expects if hasattr(result, 'expects') else None
    return total_cost, (total_cost, cost_values, terminate, expects)


def _save(
    _cost_values_over_epochs: list,
    _total_cost_over_epochs: list,
    _parameters_since_last_save: dict | list,
    _last_save_epoch: int,
    _opt_options: dict,
    _filepath: str,
):
    # don't want to resave data from the epoch we last saved at, so +1
    data_dict = {
        'cost_values': _cost_values_over_epochs[_last_save_epoch + 1 :],
        'total_cost': _total_cost_over_epochs[_last_save_epoch + 1 :],
    }
    if isinstance(_parameters_since_last_save, dict):
        data_dict |= _parameters_since_last_save
    else:
        data_dict['parameters'] = _parameters_since_last_save
    append_to_h5(_filepath, data_dict, _opt_options)


def _terminate_early(
    grads: Array | dict,
    parameters: Array | dict,
    previous_parameters: Array | dict,
    total_cost: Array,
    prev_total_cost: Array,
    terminate_for_cost: list[bool],
    epoch: int,
    opt_options: dict,
) -> None | int:
    termination_key = -1
    if epoch == opt_options['epochs'] - 1:
        termination_key = 0
    # gtol and xtol
    dx = 0.0
    dg = 0.0
    if isinstance(parameters, dict):
        for (_key_new, val_new), (_key_old, val_old) in zip(
            parameters.items(), previous_parameters.items(), strict=True
        ):
            dx += _norm(val_new - val_old)
        for _, grad_val in grads.items():
            dg += _norm(grad_val)
    else:
        dx += _norm(parameters - previous_parameters)
        dg += _norm(grads)
    if dg < opt_options['gtol']:
        termination_key = 1
    # ftol
    dF = np.abs(total_cost - prev_total_cost)
    if dF < opt_options['ftol'] * total_cost:
        termination_key = 2
    if dx < opt_options['xtol'] * (opt_options['xtol'] + dx):
        termination_key = 3
    if not opt_options['all_costs'] and any(terminate_for_cost):
        termination_key = 4
    if opt_options['all_costs'] and all(terminate_for_cost):
        termination_key = 5

    return termination_key


def _norm(x: Array) -> Array:
    if x.shape == ():
        return np.abs(x)
    return np.linalg.norm(x, ord=np.inf)
