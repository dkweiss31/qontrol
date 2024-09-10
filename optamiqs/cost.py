from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from dynamiqs import TimeArray, isdm, operator_to_vector
from dynamiqs._utils import cdtype
from dynamiqs.result import Result
from dynamiqs.time_array import SummedTimeArray
from jax import Array, vmap
from jaxtyping import ArrayLike


def _operator_to_vector(states: Array) -> Array:
    if isdm(states):
        return operator_to_vector(states)
    return states


def incoherent_infidelity(
    target_states: ArrayLike, cost_multiplier: float = 1.0
) -> IncoherentInfidelity:
    r"""Instantiate the cost function for calculating infidelity incoherently.

    This infidelity is defined as
    $$
        F_{\rm incoherent} = \sum_{k}|\langle\psi_{t}^{k}|U(\vec{\epsilon})|\psi_{i}^{k}\rangle|^2
    $$
    """  # noqa: E501
    target_states = jnp.asarray(target_states, dtype=cdtype())
    if isdm(target_states):
        target_states = operator_to_vector(target_states)
    return IncoherentInfidelity(cost_multiplier, target_states)


def coherent_infidelity(
    target_states: list[ArrayLike], cost_multiplier: float = 1.0
) -> CoherentInfidelity:
    r"""Instantiate the cost function for calculating infidelity coherently.

    This infidelity is defined as
    $$
        F_{\rm incoherent} = |\sum_{k}\langle\psi_{t}^{k}|U(\vec{\epsilon})|\psi_{i}^{k}\rangle|^2
    $$
    """  # noqa: E501
    target_states = jnp.asarray(target_states, dtype=cdtype())
    if isdm(target_states):
        target_states = operator_to_vector(target_states)
    return CoherentInfidelity(cost_multiplier, target_states)


def forbidden_states(
    forbidden_states_list: list[ArrayLike], cost_multiplier: float = 1.0
) -> ForbiddenStates:
    """Instantiate the cost function for penalizing forbidden-state occupation.

    `forbidden_states_list` should be a list of lists of forbidden states for each
    respective initial state. The resulting `forbid_array` has dimensions sfid where f
     is the batch dimension over multiple forbidden states
    """
    state_shape = _operator_to_vector(forbidden_states_list[0][0]).shape
    num_states = len(forbidden_states_list)
    num_forbid_per_state = jnp.asarray(
        [len(forbid_list) for forbid_list in forbidden_states_list]
    )
    max_num_forbid = jnp.max(num_forbid_per_state)
    arr_indices = [
        (state_idx, forbid_idx)
        for state_idx in range(num_states)
        for forbid_idx in range(max_num_forbid)
    ]
    forbid_array = jnp.zeros((num_states, max_num_forbid, *state_shape), dtype=cdtype())
    for state_idx, forbid_idx in arr_indices:
        forbidden_state = _operator_to_vector(
            forbidden_states_list[state_idx][forbid_idx]
        )
        forbid_array = forbid_array.at[state_idx, forbid_idx].set(forbidden_state)
    return ForbiddenStates(cost_multiplier, forbid_array)


def control_area(cost_multiplier: float = 1.0) -> ControlArea:
    r"""Control area cost function.

    Penalize the area under the curve according to
    $$
        C = \sum_{j}\int_{0}^{T}\Omega_{j}(t)dt,
    $$
    where the $\Omega_{j}$ are the individual controls and $T$ is the pulse time.
    """
    return ControlArea(cost_multiplier)


def control_norm(threshold: float, cost_multiplier: float = 1.0) -> ControlNorm:
    r"""Control norm cost function.

    Penalize the norm of the controls above some threshold according to
    $$
        C = \sum_{j}\int_{0}^{T}ReLU(|\Omega_{j}(t)|-\Omega_{max})dt,
    $$
    where `threshold`=$\Omega_{max}$
    """
    return ControlNorm(cost_multiplier, threshold)


def control_custom(cost_fun: callable, cost_multiplier: float = 1.0) -> ControlCustom:
    r"""Cost function based on an arbitrary transformation of the controls.

    Penalize the controls according to norm of the controls above some threshold
    according to
    $$
        C = \sum_{j}\int_{0}^{T}F(\Omega_{j}(t))dt,
    $$
    for some arbitrary function F
    """
    cost_fun = jtu.Partial(cost_fun)
    return ControlCustom(cost_multiplier, cost_fun)


def custom_cost(cost_fun: callable, cost_multiplier: float = 1.0) -> CustomCost:
    r"""A custom cost function.

    If the user has a specific cost function in mind not capture by predefined cost
    functions, they can provide a custom cost function `cost_fun` that should
    have the signature (result: Result, H: TimeArray) -> float. See
    ADVANCED API for example usages.
    """
    cost_fun = jtu.Partial(cost_fun)
    return CustomCost(cost_multiplier, cost_fun)


class Cost(eqx.Module):
    cost_multiplier: float

    def evaluate(self, result: Result, H: TimeArray) -> Array:
        raise NotImplementedError


class IncoherentInfidelity(Cost):
    target_states: Array

    def evaluate(self, result: Result, H: TimeArray) -> Array:  # noqa ARG002
        final_state = _operator_to_vector(result.final_state)
        overlaps = jnp.einsum(
            'sid,...sid->...s', jnp.conj(self.target_states), final_state
        )
        # square before summing
        overlaps_sq = jnp.real(jnp.abs(overlaps * jnp.conj(overlaps)))
        infid = 1 - jnp.mean(overlaps_sq)
        return self.cost_multiplier * infid


class CoherentInfidelity(Cost):
    target_states: Array

    def evaluate(self, result: Result, H: TimeArray) -> Array:  # noqa ARG002
        final_state = _operator_to_vector(result.final_state)
        overlaps = jnp.einsum(
            'sid,...sid->...s', jnp.conj(self.target_states), final_state
        )
        # sum before squaring
        overlaps_avg = jnp.mean(overlaps, axis=-1)
        fids = jnp.abs(overlaps_avg * jnp.conj(overlaps_avg))
        # average over any remaining batch dimensions
        infid = 1 - jnp.mean(fids)
        return self.cost_multiplier * infid


class ForbiddenStates(Cost):
    forbidden_states: Array

    def evaluate(self, result: Result, H: TimeArray) -> Array:  # noqa ARG002
        # states has dims ...stid, where s is initial_states batching, t has
        # dimension of tsave and id are the state dimensions.
        states = _operator_to_vector(result.states)
        forbidden_ovlps = jnp.einsum(
            '...stid,sfid->...stf', states, self.forbidden_states
        )
        forbidden_pops = jnp.real(jnp.mean(forbidden_ovlps * jnp.conj(forbidden_ovlps)))
        return self.cost_multiplier * forbidden_pops


class Control(Cost):
    def evaluate_controls(self, result: Result, H: TimeArray, func: callable) -> Array:
        def _evaluate_at_tsave(_H: TimeArray) -> Array:
            if hasattr(_H, 'prefactor'):
                return jnp.sum(func(vmap(_H.prefactor)(result.tsave)))
            return jnp.array(0.0)

        if isinstance(H, SummedTimeArray):
            control_val = 0.0
            # ugly for loop, having trouble with vmap or scan because only PWCTimeArray
            # and ModulatedTimeArray have attributes prefactor
            for _H in H.timearrays:
                control_val += _evaluate_at_tsave(_H)
        else:
            control_val = _evaluate_at_tsave(H)

        return self.cost_multiplier * control_val


class ControlNorm(Control):
    threshold: float

    def evaluate(self, result: Result, H: TimeArray) -> Array:
        return self.evaluate_controls(
            result, H, lambda x: jax.nn.relu(jnp.abs(x) - self.threshold)
        )


class ControlArea(Control):
    def evaluate(self, result: Result, H: TimeArray) -> Array:
        return self.evaluate_controls(result, H, lambda x: x)


class ControlCustom(Control):
    cost_fun: callable

    def evaluate(self, result: Result, H: TimeArray) -> Array:
        return self.evaluate_controls(result, H, self.cost_fun)


class CustomCost(Control):
    cost_fun: callable

    def evaluate(self, result: Result, H: TimeArray) -> Array:
        return self.cost_fun(result, H)
