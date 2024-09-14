from __future__ import annotations

import time
from functools import partial

import jax
import jax.numpy as jnp
import optax
from dynamiqs.gradient import Gradient
from dynamiqs.solver import Solver, Tsit5
from jax import Array
from jaxtyping import ArrayLike
from optax import GradientTransformation, OptState, TransformInitFn

from .cost import Cost
from .model import Model
from .options import OptimizerOptions
from .utils.file_io import save_optimization


def optimize(
    parameters: ArrayLike | dict,
    costs: list[Cost],
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
    """
    opt_state = optimizer.init(parameters)

    @partial(jax.jit, static_argnames=('_solver', '_gradient', '_options'))
    def step(
        _parameters: ArrayLike | dict,
        _costs: list[Cost],
        _model: Model,
        _opt_state: OptState,
        _solver: Solver,
        _gradient: Gradient,
        _options: OptimizerOptions,
    ) -> [Array, TransformInitFn, Array]:
        grads, _infids = jax.grad(loss, has_aux=True)(
            _parameters, _costs, _model, _solver, _gradient, _options
        )
        updates, _opt_state = optimizer.update(grads, _opt_state)
        _parameters = optax.apply_updates(_parameters, updates)
        return _parameters, _opt_state, _infids

    if options.verbose and filepath is not None:
        print(f'saving results to {filepath}')
    try:  # trick for catching keyboard interrupt
        for epoch in range(options.epochs):
            epoch_start_time = time.time()
            parameters, opt_state, infids = step(
                parameters, costs, model, opt_state, solver, gradient, options
            )
            if options.verbose:
                elapsed_time = jnp.around(time.time() - epoch_start_time, decimals=3)
                print(
                    f'epoch: {epoch}, fidelity: {1 - infids},'
                    f' elapsed_time: {elapsed_time} s'
                )
            if filepath is not None:
                save_optimization(
                    filepath,
                    {'infidelities': infids},
                    parameters,
                    options.__dict__,
                    epoch,
                )
            if all(infids < 1 - options.target_fidelity):
                print(f'target fidelity reached after {epoch} epochs')
                print(f'fidelity: {1 - infids}')
                break
            if epoch == options.epochs - 1:
                print('reached maximum number of allowed epochs')
                print(f'fidelity: {1 - infids}')
    except KeyboardInterrupt:
        print('terminated on keyboard interrupt')
    return parameters


def loss(
    parameters: Array | dict,
    costs: list[Cost],
    model: Model,
    solver: Solver,
    gradient: Gradient,
    options: OptimizerOptions,
) -> [float, Array]:
    result, H = model(parameters, solver, gradient, options)
    # manual looping here because costs is a list of classes, not straightforward to
    # call vmap on
    cost_values = [cost.evaluate(result, H) for cost in costs]
    # assumption is that the zeroth entry in costs is the infidelity,
    # which we print out so that the optimization can be monitored
    infids = cost_values[0]
    return jnp.log(jnp.sum(jnp.asarray(cost_values))), infids[None]
