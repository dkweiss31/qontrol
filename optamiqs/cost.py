from __future__ import annotations

import equinox as eqx
from jax import Array, vmap
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike

from dynamiqs._utils import cdtype
from dynamiqs import TimeArray, unit
from dynamiqs.time_array import SummedTimeArray
from dynamiqs.result import Result
from .fidelity import infidelity_incoherent, infidelity_coherent


def incoherent_infidelity(
    target_states: ArrayLike,
    cost_multiplier: float = 1.0
) -> IncoherentInfidelity:
    r"""Instantiate the cost function for calculating infidelity incoherently.

    This infidelity is defined as
    $$
        F_{\rm incoherent} = \sum_{k}|\langle\psi_{t}^{k}|U(\vec{\epsilon})|\psi_{i}^{k}\rangle|^2
    $$
    """
    target_states = jnp.asarray(target_states, dtype=cdtype())
    return IncoherentInfidelity(target_states, cost_multiplier)


class Cost(eqx.Module):
    cost_multiplier: float

    def evaluate(self, result: Result, H: TimeArray):
        raise NotImplementedError


class IncoherentInfidelity(Cost):
    target_states: Array

    def __init__(self, target_states: ArrayLike, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_states = jnp.asarray(target_states, dtype=cdtype())

    def evaluate(self, result: Result, H: TimeArray):
        final_states = result.states[..., -1, :, :]
        overlaps = jnp.einsum(
            'sid,...sid->...s', jnp.conj(self.target_states), final_states
        )
        # square before summing
        overlaps_sq = jnp.real(jnp.abs(overlaps * jnp.conj(overlaps)))
        infid = 1 - jnp.mean(overlaps_sq)
        return self.cost_multiplier * infid


class CoherentInfidelity(Cost):
    target_states: Array

    def __init__(self, target_states: ArrayLike, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_states = jnp.asarray(target_states, dtype=cdtype())

    def evaluate(self, result: Result, H: TimeArray):
        final_states = result.states[..., -1, :, :]
        overlaps = jnp.einsum(
            'sid,...sid->...s', jnp.conj(self.target_states), final_states
        )
        # sum before squaring
        overlaps_avg = jnp.mean(overlaps, axis=-1)
        fids = jnp.abs(overlaps_avg * jnp.conj(overlaps_avg))
        # average over any remaining batch dimensions
        infid = 1 - jnp.mean(fids)
        return self.cost_multiplier * infid


class ForbiddenStates(Cost):
    """
    forbidden_states should be a list of lists of forbidden states for each
    respective initial state. The resulting self.forbidden_states has
    dimensions sbid where b is the batch dimension over multiple forbidden states
    """
    forbidden_states: Array

    def __init__(self, forbidden_states: list[Array], *args, **kwargs):
        super().__init__(*args, **kwargs)
        state_shape = forbidden_states[0][0].shape
        num_states = len(forbidden_states)
        num_forbid_per_state = jnp.asarray([
            len(forbid_list) for forbid_list in forbidden_states
        ])
        max_num_forbid = jnp.max(num_forbid_per_state)
        arr_indices = [(state_idx, forbid_idx)
                       for state_idx in range(num_states)
                       for forbid_idx in range(max_num_forbid)]
        forbid_array = jnp.zeros((num_states, max_num_forbid, *state_shape), dtype=cdtype())
        for state_idx, forbid_idx in arr_indices:
            forbidden_state = forbidden_states[state_idx][forbid_idx]
            forbid_array = forbid_array.at[state_idx, forbid_idx].set(forbidden_state)
        self.forbidden_states = forbid_array

    def evaluate(self, result: Result, H: TimeArray):
        # states has dims ...stid, where s is initial_states batching, t has
        # dimension of tsave and id are the state dimensions.
        forbidden_ovlps = jnp.einsum(
            "...stid,sfid->...stf", result.states, self.forbidden_states
        )
        forbidden_pops = jnp.real(jnp.mean(forbidden_ovlps * jnp.conj(forbidden_ovlps)))
        return self.cost_multiplier * forbidden_pops


class Control(Cost):

    def __init__(self, *args, **kwargs):
        # TODO allow weighting for different Hamiltonian controls
        super().__init__(*args, **kwargs)

    def evaluate_controls(self, result: Result, H: TimeArray, func):

        def _evaluate_at_tsave(_H):
            if hasattr(_H, "prefactor"):
                return jnp.sum(func(vmap(_H.prefactor)(result.tsave)))
            else:
                return jnp.array(0.0)

        if isinstance(H, SummedTimeArray):
            control_val = 0.0
            # ugly for loop, having trouble with vmap or scan because only PWCTimeArray
            # and ModulatedTimeArray have attributes prefactor
            for timearray in H.timearrays:
                control_val += jnp.sum(vmap(_evaluate_at_tsave)(timearray))
        else:
            control_val = _evaluate_at_tsave(H)

        return self.cost_multiplier * control_val


class ControlNorm(Control):

    def evaluate(self, result: Result, H: TimeArray):
        return self.evaluate_controls(result, H, lambda x: x ** 2)


class ControlArea(Control):

    def evaluate(self, result: Result, H: TimeArray):
        return self.evaluate_controls(result, H, lambda x: x)


class MCInfidelity(Cost):
    target_states: ArrayLike
    target_states_traj: ArrayLike
    coherent: bool
    no_jump_weight: float
    jump_weight: float
    infid_func: callable

    def __init__(
        self,
        target_states,
        target_states_traj,
        coherent=False,
        no_jump_weight=1.0,
        jump_weight=1.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.target_states = jnp.asarray(target_states, dtype=cdtype())
        self.target_states_traj = jnp.asarray(target_states_traj, dtype=cdtype())
        if coherent:
            self.infid_func = jax.tree_util.Partial(infidelity_coherent)
        else:
            self.infid_func = jax.tree_util.Partial(infidelity_incoherent)
        self.coherent = coherent
        self.no_jump_weight = no_jump_weight
        self.jump_weight = jump_weight

    def evaluate(self, result: Result, H: TimeArray):
        final_jump_states = unit(result.final_jump_states).swapaxes(-4, -3)
        final_no_jump_states = unit(result.final_no_jump_state)
        infids_jump = self.infid_func(
            final_jump_states, self.target_states_traj
        )
        infids_no_jump = self.infid_func(
            final_no_jump_states, self.target_states
        )
        infid = (self.jump_weight * jnp.mean(infids_jump)
                 + self.no_jump_weight * jnp.mean(infids_no_jump))
        return self.cost_multiplier * infid


class CustumCost(Control):
    cost_fun: callable

    def __init__(self, cost_fun: callable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cost_fun = jax.tree_util.Partial(cost_fun)

    def evaluate(self, result: Result, H: TimeArray):
        return self.cost_fun(result, H)
