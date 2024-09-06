from __future__ import annotations

import time
from functools import partial

import dynamiqs as dq
import jax
import jax.numpy as jnp
import optax
from dynamiqs._utils import cdtype
from dynamiqs.solver import Solver, Tsit5
from jax import Array
from jaxtyping import ArrayLike
from optax import GradientTransformation, TransformInitFn
from jaxtyping import ArrayLike
from dynamiqs.integrators._utils import _astimearray
from jax.random import PRNGKey

from .file_io import save_and_print
from .options import GRAPEOptions
from .pulse_optimizer import PulseOptimizer
from .cost import Cost


def grape(
    hamiltonian_time_update: HamiltonianTimeUpdater,
    initial_states: ArrayLike,
    params_to_optimize: ArrayLike,
    *,
    costs: list[Cost] = None,
    jump_ops: list[ArrayLike] = None,
    exp_ops: list[ArrayLike] = None,
    filepath: str = 'tmp.h5py',
    optimizer: GradientTransformation = optax.adam(0.1, b1=0.99, b2=0.99),  # noqa: B008
    solver: Solver = Tsit5(),  # noqa: B008
    options: GRAPEOptions = GRAPEOptions(),  # noqa: B008
    init_params_to_save: dict | None = None,
) -> Array:
    r"""Perform gradient descent to optimize Hamiltonian parameters.

    This function takes as input a list of initial_states and a list of
    target_states, and optimizes params_to_optimize to achieve the highest fidelity
    state transfer. It saves the parameters from every epoch and the associated fidelity
    in the file filepath

    Args:
         H_func _(PyTree object)_: Hamiltonian. Assumption is that we can
            instantiate a timecallable instance with
            H_func = partial(H_func, drive_params=params_to_optimize)
            H = timecallable(H_func, )
         initial_states _(list of array-like of shape (n, 1))_: initial states
         target_states _(list of array-like of shape (n, 1))_: target states
         tsave _(array-like of shape (nt,))_: times to be passed to sesolve
         params_to_optimize _(dict or array-like)_: parameters to optimize
            over that are used to define the Hamiltonian
         filepath _(str)_: filepath of where to save optimization results
         optimizer _(optax.GradientTransformation)_: optax optimizer to use
            for gradient descent. Defaults to the Adam optimizer
         solver _(Solver)_: solver passed to sesolve
         options _(Options)_: options for grape optimization and sesolve integration
            relevant options include:
                coherent, bool where if True we use a definition of fidelity
                that includes relative phases, if not it ignores relative phases
                epochs, int that is the maximum number of epochs to loop over
                target_fidelity, float where the optimization terminates if the fidelity
                if above this value
         init_params_to_save _(dict)_: initial parameters we want to save
    Returns:
        optimized parameters from the final timestep
    """
    if init_params_to_save is None:
        init_params_to_save = {}
    if not options.save_states:
        raise ValueError(
            "save_states must be set to true for GRAPE optimization. If you are worried"
            "about excess memory usage, you can specify that the tsave passed to sesolve"
            "only has two elements in it, e.g. tsave=(0, t). That is to say, the control"
            "tsave need not be the same as the tsave for saving states."
        )
    initial_states = jnp.asarray(initial_states, dtype=cdtype())
    if jump_ops is not None:
        jump_ops = [_astimearray(L) for L in jump_ops]
    exp_ops = jnp.asarray(exp_ops, dtype=cdtype()) if exp_ops is not None else None
    opt_state = optimizer.init(params_to_optimize)
    _, init_tsave = hamiltonian_time_update.update(params_to_optimize)
    init_param_dict = options.__dict__ | {'tsave': init_tsave} | init_params_to_save
    print(f'saving results to {filepath}')
    try:  # trick for catching keyboard interrupt
        for epoch in range(options.epochs):
            epoch_start_time = time.time()
            params_to_optimize, opt_state, infids = step(
                params_to_optimize,
                opt_state,
                hamiltonian_time_update,
                initial_states,
                costs,
                jump_ops,
                exp_ops,
                solver,
                options,
                optimizer,
            )
            data_dict = {'infidelities': infids}
            save_and_print(
                filepath,
                data_dict,
                params_to_optimize,
                init_param_dict,
                epoch,
                epoch_start_time,
            )
            if all(infids < 1 - options.target_fidelity):
                print('target fidelity reached')
                break
    except KeyboardInterrupt:
        print('terminated on keyboard interrupt')
    else:
        print('reached maximum number of allowed epochs')
    print(f'all results saved to {filepath}')
    return params_to_optimize


@partial(jax.jit, static_argnames=('solver', 'options', 'optimizer'))
def step(
    params_to_optimize: Array,
    opt_state: TransformInitFn,
    hamiltonian_time_update: HamiltonianTimeUpdater,
    initial_states: Array,
    costs: list[Cost],
    jump_ops: list[Array],
    exp_ops: list[Array],
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
        hamiltonian_time_update,
        initial_states,
        costs,
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
    hamiltonian_time_update: HamiltonianTimeUpdater,
    initial_states: Array,
    costs: list[Cost],
    jump_ops: Array,
    exp_ops: Array,
    solver: Solver,
    options: GRAPEOptions,
) -> [float, Array]:
    H, tsave = hamiltonian_time_update.update(params_to_optimize)
    if options.grape_type == 0:
        results = dq.sesolve(H, initial_states, tsave, solver=solver, options=options)
    elif options.grape_type == 1:
        results = dq.mesolve(
            H,
            jump_ops,
            initial_states,
            tsave,
            exp_ops=exp_ops,
            solver=solver,
            options=options
        )
    elif options.grape_type == 2:
        results = dq.mcsolve(
            H,
            jump_ops,
            initial_states,
            tsave,
            key=PRNGKey(options.rng_seed),
            exp_ops=exp_ops,
            solver=solver,
            options=options,
        )
    else:
        raise ValueError(
            f"grape_type can be 'sesolve', 'mesolve', or 'mcsolve' but got"
            f"{options.grape_type}"
        )
    # manual looping here becuase costs is a list of classes, not straightforward to
    # call vmap on
    cost_values = [cost.evaluate(results, H) for cost in costs]
    # assumption is that the zeroth entry in costs is the infidelity,
    # which we print out so that the optimization can be monitored
    infids = cost_values[0]
    return jnp.log(jnp.sum(jnp.asarray(cost_values))), infids[None]
