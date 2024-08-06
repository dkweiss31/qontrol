from __future__ import annotations

import time
from functools import partial

import jax
import jax.numpy as jnp
import optax
from jaxtyping import ArrayLike

import dynamiqs as dq
from dynamiqs import Options
from .fidelity import infidelity_coherent, infidelity_incoherent
from .file_io import save_and_print

from dynamiqs._utils import cdtype
from dynamiqs.solver import Solver, Tsit5
from dynamiqs.time_array import timecallable, BatchedCallable

__all__ = ["grape"]


def grape(
    H_func: BatchedCallable,
    initial_states: ArrayLike,
    target_states: ArrayLike,
    tsave: ArrayLike,
    params_to_optimize: ArrayLike,
    *,
    filepath: str = "tmp.h5py",
    optimizer: optax.GradientTransformation = optax.adam(0.1, b1=0.99, b2=0.99),
    solver: Solver = Tsit5(),
    options: Options = Options(),
    init_params_to_save: dict = {},
) -> ArrayLike:
    r"""Perform gradient descent to optimize Hamiltonian parameters

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
    initial_states = jnp.asarray(initial_states, dtype=cdtype())
    target_states = jnp.asarray(target_states, dtype=cdtype())
    opt_state = optimizer.init(params_to_optimize)
    init_param_dict = options.__dict__ | {"tsave": tsave} | init_params_to_save
    print(f"saving results to {filepath}")
    try:  # trick for catching keyboard interrupt
        for epoch in range(options.epochs):
            epoch_start_time = time.time()
            params_to_optimize, opt_state, infids = step(
                params_to_optimize,
                opt_state,
                H_func,
                initial_states,
                target_states,
                tsave,
                solver,
                options,
                optimizer,
            )
            data_dict = {"infidelities": infids}
            save_and_print(
                filepath,
                data_dict,
                params_to_optimize,
                init_param_dict,
                epoch,
                epoch_start_time,
            )
            if all(infids < 1 - options.target_fidelity):
                print("target fidelity reached")
                break
        print(f"all results saved to {filepath}")
        return params_to_optimize
    except KeyboardInterrupt:
        print("terminated on keyboard interrupt")
        print(f"all results saved to {filepath}")
        return params_to_optimize


@partial(jax.jit, static_argnames=("solver", "options", "optimizer"))
def step(
    params_to_optimize,
    opt_state,
    H_func,
    initial_states,
    target_states,
    tsave,
    solver,
    options,
    optimizer,
):
    """calculate gradient of the loss and step updated parameters.
    We have has_aux=True because loss also returns the infidelities on the side
    (want to save those numbers as they give info on which pulse was best)"""
    grads, infids = jax.grad(loss, has_aux=True)(
        params_to_optimize,
        H_func,
        initial_states,
        target_states,
        tsave,
        solver,
        options,
    )
    updates, opt_state = optimizer.update(grads, opt_state)
    params_to_optimize = optax.apply_updates(params_to_optimize, updates)
    return params_to_optimize, opt_state, infids


def loss(
    params_to_optimize,
    H_func,
    initial_states,
    target_states,
    tsave,
    solver,
    options,
):
    if options.coherent:
        infid_func = infidelity_coherent
    else:
        infid_func = infidelity_incoherent
    H_func = partial(H_func, drive_params=params_to_optimize)
    H = timecallable(
        H_func,
    )
    results = dq.sesolve(H, initial_states, tsave, solver=solver, options=options)
    if options.save_states:
        final_states = results.states[..., -1, :, :]
    else:
        final_states = results.states
    infids = infid_func(final_states, target_states)
    if infids.ndim == 0:
        # for saving purposes, want this to be an Array as opposed to a float
        infids = infids[None]
    return jnp.log(jnp.sum(infids)), infids
