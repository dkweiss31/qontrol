import equinox as eqx
import jax.debug
from jax import Array
import jax.numpy as jnp
from jaxtyping import ArrayLike

from dynamiqs._utils import cdtype
from dynamiqs import TimeArray
from .fidelity import infidelity_incoherent, infidelity_coherent


class Cost(eqx.Module):
    cost_multiplier: float = 1.0

    def evaluate(self, states: Array, final_states: Array, time_array: TimeArray):
        raise NotImplementedError


class IncoherentInfidelity(Cost):
    target_states: Array

    def __init__(self, target_states: ArrayLike, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_states = jnp.asarray(target_states, dtype=cdtype())

    def evaluate(self, states: Array, final_states: Array, time_array: TimeArray):
        return self.cost_multiplier * infidelity_incoherent(
            final_states, self.target_states, average=True
        )[None]


class CoherentInfidelity(Cost):
    target_states: Array

    def __init__(self, target_states: ArrayLike, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_states = jnp.asarray(target_states, dtype=cdtype())

    def evaluate(self, states: Array, final_states: Array, time_array: TimeArray):
        infid = infidelity_coherent(final_states, self.target_states)
        return self.cost_multiplier * jnp.average(infid)[None]


class ForbiddenStates(Cost):
    """
    forbidden_states should be a list of lists of forbidden states for each
    respective initial state.
    """
    forbidden_states: list[Array]

    def __init__(self, forbidden_states: list[Array], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forbidden_states = [jnp.asarray(_forbidden_states, dtype=cdtype())
                                 for _forbidden_states in forbidden_states]

    def evaluate(self, states: Array, final_states: Array, time_array: TimeArray):
        # states has dims ...stid, where s is initial_states batching, t has
        # dimension of tsave and id are the state dimensions
        states = jnp.moveaxis(states, -4, 0)
        forbidden_pops = 0.0
        for state_idx, state in enumerate(states):
            unforbidden_pops = infidelity_incoherent(
                state, self.forbidden_states[state_idx], average=False
            )
            forbidden_pops += jnp.mean(1 - unforbidden_pops)
        return self.cost_multiplier * forbidden_pops[None]
