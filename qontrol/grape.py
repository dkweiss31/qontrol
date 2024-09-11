from __future__ import annotations

import time
from functools import partial

import dynamiqs as dq
import jax
import jax.numpy as jnp
import optax
from dynamiqs._utils import cdtype
from dynamiqs.integrators._utils import _astimearray
from dynamiqs.solver import Solver, Tsit5
from jax import Array
from jaxtyping import ArrayLike
from optax import GradientTransformation, TransformInitFn

from .cost import Cost
from .file_io import save_optimization
from .options import GRAPEOptions
from .update import Updater, updater

SEGRAPE = 0
MEGRAPE = 1


def grape(
    params_to_optimize: ArrayLike,
    costs: list[Cost],
    hamiltonian_updater: Updater,
    initial_states: ArrayLike,
    *,
    tsave: ArrayLike | None = None,
    tsave_updater: Updater | None = None,
    jump_ops: ArrayLike | None = None,
    exp_ops: ArrayLike | None = None,
    filepath: str | None = None,
    optimizer: GradientTransformation = optax.adam(0.0001, b1=0.99, b2=0.99),  # noqa: B008
    solver: Solver = Tsit5(),  # noqa: B008
    options: GRAPEOptions = GRAPEOptions(),  # noqa: B008
    init_params_to_save: dict | None = None,
) -> Array | dict:
    r"""Perform gradient descent to optimize Hamiltonian parameters.

    This function takes as input a list of initial_states and a list of
    target_states, and optimizes params_to_optimize to achieve the highest fidelity
    state transfer. It saves the parameters from every epoch and the associated fidelity
    in the file filepath

    Args:
        params_to_optimize _(dict or array-like)_: parameters to optimize
            over that are used to define the Hamiltonian and control times.
        costs _(list of Cost instances)_: List of cost functions used to perform the
            optimization.
        hamiltonian_updater _(Updater)_: Class specifying
            how to update the Hamiltonian and control times.
        initial_states _(list of array-like of shape (n, 1) or (n, n))_: Initial states.
        tsave _(array like)_: Times passed to dynamiqs sesolve or mesolve
        tsave_updater _(Updater)_: Updater for tsave that adjusts the control times for
            time-optimal control. One of tsave or tsave_updater need to be passed, but
            not both.
        jump_ops _(list of array-like)_: Jump operators to use if performing mesolve
            optimizations, not utilized if performing sesolve optimizations.
        exp_ops _(list of array-like)_: Operators to calculate expectation values of,
            in case some of the cost functions depend on the value of certain
            expectation values.
        filepath _(str)_: Filepath of where to save optimization results.
        optimizer _(optax.GradientTransformation)_: optax optimizer to use
            for gradient descent. Defaults to the Adam optimizer.
        solver _(Solver)_: Solver passed to dynamiqs.
        options _(GRAPEOptions)_: Options for grape optimization and dynamiqs
            integration.
        init_params_to_save _(dict)_: Initial parameters we want to save.

    Returns:
        optimized parameters from the final timestep
    """
    if init_params_to_save is None:
        init_params_to_save = {}
    initial_states = jnp.asarray(initial_states, dtype=cdtype())
    if jump_ops is not None:
        jump_ops = [_astimearray(L) for L in jump_ops]
    exp_ops = jnp.asarray(exp_ops, dtype=cdtype()) if exp_ops is not None else None

    if tsave is None and tsave_updater is None:
        raise ValueError('one of tsave or tsave_updater need to not be None')
    if tsave is not None and tsave_updater is not None:
        raise ValueError(
            'only one of tsave or tsave_updater can be set but got'
            f'{tsave} for tsave and {tsave_updater} for tsave_updater'
        )
    if tsave_updater is None:
        tsave_updater = updater(lambda _: jnp.asarray(tsave))
    init_tsave = tsave_updater.update(params_to_optimize)
    init_param_dict = options.__dict__ | {'tsave': init_tsave} | init_params_to_save

    opt_state = optimizer.init(params_to_optimize)

    if options.verbose and filepath is not None:
        print(f'saving results to {filepath}')
    try:  # trick for catching keyboard interrupt
        for epoch in range(options.epochs):
            epoch_start_time = time.time()
            params_to_optimize, opt_state, infids = step(
                params_to_optimize,
                costs,
                hamiltonian_updater,
                initial_states,
                tsave_updater,
                opt_state,
                jump_ops,
                exp_ops,
                solver,
                options,
                optimizer,
            )
            data_dict = {'infidelities': infids}
            if options.verbose:
                elapsed_time = jnp.around(time.time() - epoch_start_time, decimals=3)
                print(
                    f'epoch: {epoch}, fidelity: {1 - infids},'
                    f' elapsed_time: {elapsed_time} s'
                )
            if filepath is not None:
                save_optimization(
                    filepath, data_dict, params_to_optimize, init_param_dict, epoch
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
    return params_to_optimize


@partial(jax.jit, static_argnames=('solver', 'options', 'optimizer'))
def step(
    params_to_optimize: Array,
    costs: list[Cost],
    hamiltonian_updater: Updater,
    initial_states: Array,
    tsave_updater: Updater,
    opt_state: TransformInitFn,
    jump_ops: list[dq.TimeArray],
    exp_ops: Array,
    solver: Solver,
    options: GRAPEOptions,
    optimizer: GradientTransformation,
) -> [Array, TransformInitFn, Array]:
    """Calculate gradient of the loss and step updated parameters.

    We have has_aux=True because loss also returns the infidelities on the side
    (want to save those numbers as they give info on which pulse was best).
    """
    grads, infids = jax.grad(loss, has_aux=True)(
        params_to_optimize,
        costs,
        hamiltonian_updater,
        initial_states,
        tsave_updater,
        jump_ops,
        exp_ops,
        solver,
        options,
    )
    updates, opt_state = optimizer.update(grads, opt_state)
    params_to_optimize = optax.apply_updates(params_to_optimize, updates)
    return params_to_optimize, opt_state, infids


def loss(
    params_to_optimize: Array,
    costs: list[Cost],
    hamiltonian_updater: Updater,
    initial_states: Array,
    tsave_updater: Updater,
    jump_ops: Array,
    exp_ops: Array,
    solver: Solver,
    options: GRAPEOptions,
) -> [float, Array]:
    H = hamiltonian_updater.update(params_to_optimize)
    tsave = tsave_updater.update(params_to_optimize)
    if options.grape_type == SEGRAPE:
        results = dq.sesolve(H, initial_states, tsave, solver=solver, options=options)
    elif options.grape_type == MEGRAPE:
        results = dq.mesolve(
            H,
            jump_ops,
            initial_states,
            tsave,
            exp_ops=exp_ops,
            solver=solver,
            options=options,
        )
    else:
        raise ValueError(
            f"grape_type can be 'sesolve' or 'mesolve' but got" f'{options.grape_type}'
        )
    # manual looping here because costs is a list of classes, not straightforward to
    # call vmap on
    cost_values = [cost.evaluate(results, H) for cost in costs]
    # assumption is that the zeroth entry in costs is the infidelity,
    # which we print out so that the optimization can be monitored
    infids = cost_values[0]
    return jnp.log(jnp.sum(jnp.asarray(cost_values))), infids[None]
