from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from dynamiqs import asqarray, isket, QArray, QArrayLike, TimeQArray
from dynamiqs.result import SolveResult
from dynamiqs.time_qarray import SummedTimeQArray
from jax import Array, vmap


def incoherent_infidelity(
    target_states: list[QArrayLike],
    cost_multiplier: float = 1.0,
    target_cost: float = 0.005,
) -> IncoherentInfidelity:
    r"""Instantiate the cost function for calculating infidelity incoherently.

    This infidelity is defined as
    $$
        F_{\rm incoherent} = \sum_{k}|\langle\psi_{t}^{k}|\psi_{i}^{k}(T)\rangle|^2,
    $$
    where the states at the end of the pulse are $|\psi_{i}^{k}(T)\rangle$ and the
    target states are $|\psi_{t}^{k}\rangle$.

    Args:
        target_states _(qarray_like of shape (s, n, 1) or (s, n, n))_: target states for
            the initial states passed to `grape`. If performing master-equation
            optimization, the target states should be passed as a list of density matrices.
        cost_multiplier _(float)_: Weight for this cost function relative to other cost
            functions.
        target_cost _(float)_: Target value for this cost function. If options.all_costs
            is True, the optimization terminates early if all cost functions fall below
            their target values. If options.all_costs is False, the optimization
            terminates if only one cost function falls below its target value.

    Returns:
        _(IncoherentInfidelity)_: Callable object that returns the incoherent infidelity
            and whether the infidelity is below the target value.
    """  # noqa: E501
    return IncoherentInfidelity(cost_multiplier, target_cost, asqarray(target_states))


def coherent_infidelity(
    target_states: list[QArrayLike],
    cost_multiplier: float = 1.0,
    target_cost: float = 0.005,
) -> CoherentInfidelity:
    r"""Instantiate the cost function for calculating infidelity coherently.

    This infidelity is defined as
    $$
        F_{\rm coherent} = |\sum_{k}\langle\psi_{t}^{k}|\psi_{i}^{k}(T)\rangle|^2,
    $$
    where the states at the end of the pulse are $|\psi_{i}^{k}(T)\rangle$ and the
    target states are $|\psi_{t}^{k}\rangle$.

    Args:
        target_states _(qarray_like of shape (s, n, 1) or (s, n, n))_: target states for
            the initial states passed to `grape`. If performing master-equation
            optimization, the target states should be passed as a list of density matrices.
        cost_multiplier _(float)_: Weight for this cost function relative to other cost
            functions.
        target_cost _(float)_: Target value for this cost function. If options.all_costs
            is True, the optimization terminates early if all cost functions fall below
            their target values. If options.all_costs is False, the optimization
            terminates if only one cost function falls below its target value.

    Returns:
        _(CoherentInfidelity)_: Callable object that returns the coherent infidelity
            and whether the infidelity is below the target value.
    """  # noqa: E501
    return CoherentInfidelity(cost_multiplier, target_cost, asqarray(target_states))


def forbidden_states(
    forbidden_states_list: list[QArrayLike],
    cost_multiplier: float = 1.0,
    target_cost: float = 0.0,
) -> ForbiddenStates:
    r"""Instantiate the cost function for penalizing forbidden-state occupation.

    This cost function is defined as
    $$
        C = \sum_{k}\sum_{f}\int_{0}^{T}dt|\langle\psi_{f}^{k}|\psi_{i}^{k}(t)\rangle|^2,
    $$
    where $|\psi_{i}^{k}(t)\rangle$ is the $k^{\rm th}$ initial state propagated to time
    $t$ and |\psi_{f}^{k}\rangle is a forbidden state for the $k^{\rm th}$ initial
    state.

    Args:
        forbidden_states_list _(list of list of qarray-like of shape (n, 1) or (n, n))_:
            For each initial state indexed by s (outer list), a list of forbidden states
            (inner list) should be provided. The inner lists need not all be of the same shape,
            for instance if some initial states have more forbidden states than others. The array
            is eventually reshaped to have shape (s, 1, f, n, 1) or (s, 1, f, n, n) (for
            `sesolve` or `mesolve`, respectively) where s is the number of initial
            states, f is the length of the longest forbidden-state list (with
            zero-padding for meaningless entries) and 1 is an added batch dimension for
            eventually batching over tsave.
        cost_multiplier _(float)_: Weight for this cost function relative to other cost
            functions.
        target_cost _(float)_: Target value for this cost function. If options.all_costs
            is True, the optimization terminates early if all cost functions fall below
            their target values. If options.all_costs is False, the optimization
            terminates if only one cost function falls below its target value.

    Returns:
        _(ForbiddenStates)_: Callable object that returns the forbidden-state cost
            and whether the cost is below the target value.
    """  # noqa: E501
    state_shape = forbidden_states_list[0][0].shape
    num_states = len(forbidden_states_list)  # should be the number of initial states
    num_forbid_per_state = jnp.asarray(
        [len(forbid_list) for forbid_list in forbidden_states_list]
    )
    max_num_forbid = jnp.max(num_forbid_per_state)
    arr_indices = [
        (state_idx, forbid_idx)
        for state_idx in range(num_states)
        for forbid_idx in range(num_forbid_per_state[state_idx])
    ]
    # add in a dimension for tsave that will be broadcast with the final states
    forbid_array = jnp.zeros((num_states, 1, max_num_forbid, *state_shape))
    for state_idx, forbid_idx in arr_indices:
        forbidden_state = asqarray(
            forbidden_states_list[state_idx][forbid_idx]
        ).to_jax()  # TODO fix sparse to dense conversion here
        forbid_array = forbid_array.at[state_idx, 0, forbid_idx].set(forbidden_state)
    return ForbiddenStates(cost_multiplier, target_cost, asqarray(forbid_array))


def control_area(
    cost_multiplier: float = 1.0, target_cost: float = 0.0
) -> ControlCostArea:
    r"""Control area cost function.

    Penalize the area under the pulse curves according to
    $$
        C = \sum_{j}\int_{0}^{T}\Omega_{j}(t)dt,
    $$
    where the $\Omega_{j}$ are the individual controls and $T$ is the pulse time.

    Args:
        cost_multiplier _(float)_: Weight for this cost function relative to other cost
            functions.
        target_cost _(float)_: Target value for this cost function. If options.all_costs
            is True, the optimization terminates early if all cost functions fall below
            their target values. If options.all_costs is False, the optimization
            terminates if only one cost function falls below its target value.

    Returns:
        _(ControlArea)_: Callable object that returns the control-area cost
            and whether the cost is below the target value.
    """
    return ControlCostArea(cost_multiplier, target_cost)


def control_norm(
    threshold: float, cost_multiplier: float = 1.0, target_cost: float = 0.0
) -> ControlCostNorm:
    r"""Control norm cost function.

    Penalize the norm of the controls above some threshold according to
    $$
        C = \sum_{j}\int_{0}^{T}ReLU(|\Omega_{j}(t)|-\Omega_{max})dt,
    $$
    where the $\Omega_{j}$ are the individual controls, $T$ is the pulse time
    and $\Omega_{max}$ is the threshold.

    Args:
        threshold _(float)_: Threshold to use for penalizing amplitudes above this value
            in absolute magnitude.
        cost_multiplier _(float)_: Weight for this cost function relative to other cost
            functions.
        target_cost _(float)_: Target value for this cost function. If options.all_costs
            is True, the optimization terminates early if all cost functions fall below
            their target values. If options.all_costs is False, the optimization
            terminates if only one cost function falls below its target value.

    Returns:
        _(ControlNorm)_: Callable object that returns the control-norm cost
            and whether the cost is below the target value.
    """
    return ControlCostNorm(cost_multiplier, target_cost, threshold)


def custom_control_cost(
    cost_fun: callable, cost_multiplier: float = 1.0, target_cost: float = 0.0
) -> CustomControlCost:
    r"""Cost function based on an arbitrary transformation of the controls.

    Penalize the controls according to an arbitrary function `F`
    $$
        C = \sum_{j}\int_{0}^{T}F(\Omega_{j}(t))dt,
    $$

    Args:
        cost_fun _(callable)_: Cost function which must have signature `(control_amp: Array) -> Array`.
        cost_multiplier _(float)_: Weight for this cost function relative to other cost
            functions.
        target_cost _(float)_: Target value for this cost function. If options.all_costs
            is True, the optimization terminates early if all cost functions fall below
            their target values. If options.all_costs is False, the optimization
            terminates if only one cost function falls below its target value.

    Returns:
        _(CustomCost)_: Callable object that returns the cost for the custom function
            and whether the cost is below the target value.

    Examples:
        ```python
        def penalize_negative(control_amp: jax.Array) -> jax.Array:
            return jax.nn.relu(-control_amp)


        negative_amp_cost = ql.custom_control_cost(penalize_negative)
        ```
        In this example, we penalize negative drive amplitudes.
    """  # noqa: E501
    cost_fun = jtu.Partial(cost_fun)
    return CustomControlCost(cost_multiplier, target_cost, cost_fun)


def custom_cost(
    cost_fun: callable, cost_multiplier: float = 1.0, target_cost: float = 0.0
) -> CustomCost:
    r"""A custom cost function.

    In many (most!) cases, the user may want to add a cost function to their
    optimization that is not included in the hardcoded set of available cost functions.

    Args:
        cost_fun _(callable)_: Cost function which must have signature
            `(result: dq.SolveResult, H: dq.TimeQArray, parameters: dict | Array) -> Array`.
        cost_multiplier _(float)_: Weight for this cost function relative to other cost
            functions.
        target_cost _(float)_: Target value for this cost function. If options.all_costs
            is True, the optimization terminates early if all cost functions fall below
            their target values. If options.all_costs is False, the optimization
            terminates if only one cost function falls below its target value.

    Returns:
        _(CustomCost)_: Callable object that returns the cost for the custom function.

    Examples:
        Let's imagine we want to penalize the value of some expectation value at the final
        time in `tsave`.

        ```python
        def penalize_expect(
            result: SolveResult, H: TimeQArray, parameters: dict | Array
        ) -> Array:
            # 0 is the index of the operator, -1 is the time index
            return jnp.sum(jnp.abs(result.expects[0, -1]))


        expect_cost = ql.custom_cost(penalize_expect)
        ```
        Then `expect_cost` can be added to the other utilized cost functions. The only
        thing happening under the hood is that the `penalize_expect` function is passed
        to `jax.tree_util.Partial` to enable it to be passed through jitted functions
        without issue.
    """  # noqa: E501
    cost_fun = jtu.Partial(cost_fun)
    return CustomCost(cost_multiplier, target_cost, cost_fun)


class Cost(eqx.Module):
    def __call__(
        self, result: SolveResult, H: TimeQArray, parameters: dict | Array
    ) -> Array:
        raise NotImplementedError

    def __add__(self, other: Cost) -> SummedCost:
        if isinstance(other, Cost):
            return SummedCost([self, other])
        raise NotImplementedError

    def __mul__(self, other: float) -> Cost:
        pass

    def __rmul__(self, other: float) -> Cost:
        return self * other

    def __repr__(self) -> str:
        return type(self).__name__


class SummedCost(Cost):
    costs: list[Cost]

    def __call__(
        self, result: SolveResult, H: TimeQArray, parameters: dict | Array
    ) -> list[Array]:
        return [cost(result, H, parameters)[0] for cost in self.costs]

    def __mul__(self, y: float) -> SummedCost:
        costs = [cost * y for cost in self.costs]
        return SummedCost(costs)

    def __add__(self, other: Cost) -> SummedCost:
        if isinstance(other, Cost):
            return SummedCost([*self.costs, other])
        raise NotImplementedError


class IncoherentInfidelity(Cost):
    cost_multiplier: float
    target_cost: float
    target_states: QArray

    def __call__(
        self,
        result: SolveResult,
        H: TimeQArray,  # noqa ARG002
        parameters: dict | Array,  # noqa ARG002
    ) -> tuple[tuple[Array, Array]]:
        overlaps = self.target_states.dag() @ result.final_state
        if not isket(result.final_state):
            overlaps = overlaps.trace()
        # square before summing
        infid = 1 - jnp.mean(jnp.abs(overlaps * jnp.conj(overlaps)))
        cost = self.cost_multiplier * infid
        return ((cost, cost < self.target_cost),)

    def __mul__(self, other: float) -> IncoherentInfidelity:
        return IncoherentInfidelity(
            other * self.cost_multiplier, self.target_cost, self.target_states
        )


class CoherentInfidelity(Cost):
    cost_multiplier: float
    target_cost: float
    target_states: QArray

    def __call__(
        self,
        result: SolveResult,
        H: TimeQArray,  # noqa ARG002
        parameters: dict | Array,  # noqa ARG002
    ) -> tuple[tuple[Array, Array]]:
        overlaps = self.target_states.dag() @ result.final_state
        if not isket(result.final_state):
            overlaps = overlaps.trace()
        # average over states before squaring
        overlaps_avg = jnp.mean(jnp.squeeze(overlaps), axis=-1)
        # average over any remaining batch dimensions
        infid = 1 - jnp.mean(jnp.abs(overlaps_avg * jnp.conj(overlaps_avg)))
        cost = self.cost_multiplier * infid
        return ((cost, cost < self.target_cost),)

    def __mul__(self, other: float) -> CoherentInfidelity:
        return CoherentInfidelity(
            other * self.cost_multiplier, self.target_cost, self.target_states
        )


class ForbiddenStates(Cost):
    cost_multiplier: float
    target_cost: float
    forbidden_states: QArray

    def __call__(
        self,
        result: SolveResult,
        H: TimeQArray,  # noqa ARG002
        parameters: dict | Array,  # noqa ARG002
    ) -> tuple[tuple[Array, Array]]:
        # states has dims ...stid, where s is initial_states batching, t has dimension
        # of tsave and id are the state dimensions. Want it to be stfid
        states = result.states[..., None, :, :]
        forbidden_ovlps = states.dag() @ self.forbidden_states
        if not isket(result.states):
            forbidden_ovlps = forbidden_ovlps.trace()
        forbidden_pops = jnp.real(jnp.sum(forbidden_ovlps * jnp.conj(forbidden_ovlps)))
        cost = self.cost_multiplier * forbidden_pops
        return ((cost, cost < self.target_cost),)

    def __mul__(self, other: float) -> ForbiddenStates:
        return ForbiddenStates(
            other * self.cost_multiplier, self.target_cost, self.forbidden_states
        )


class ControlCost(Cost):
    cost_multiplier: float
    target_cost: float

    def evaluate_controls(
        self, result: SolveResult, H: TimeQArray, func: callable
    ) -> Array:
        def _evaluate_at_tsave(_H: TimeQArray) -> Array:
            if hasattr(_H, 'prefactor'):
                return jnp.sum(func(vmap(_H.prefactor)(result.tsave)))
            return jnp.array(0.0)

        if isinstance(H, SummedTimeQArray):
            control_val = 0.0
            # ugly for loop, having trouble with vmap or scan because only PWCTimeQArray
            # and ModulatedTimeQArray have attributes prefactor
            for _H in H.timeqarrays:
                control_val += _evaluate_at_tsave(_H)
        else:
            control_val = _evaluate_at_tsave(H)

        return self.cost_multiplier * control_val

    def __mul__(self, other: float) -> ControlCost:
        return ControlCost(other * self.cost_multiplier, self.target_cost)


class ControlCostNorm(ControlCost):
    threshold: float

    def __call__(
        self,
        result: SolveResult,
        H: TimeQArray,
        parameters: dict | Array,  # noqa ARG002
    ) -> tuple[tuple[Array, Array]]:
        cost = jnp.abs(
            self.evaluate_controls(
                result, H, lambda x: jax.nn.relu(jnp.abs(x) - self.threshold)
            )
        )
        return ((cost, cost < self.target_cost),)

    def __mul__(self, other: float) -> ControlCostNorm:
        return ControlCostNorm(
            other * self.cost_multiplier, self.target_cost, self.threshold
        )


class ControlCostArea(ControlCost):
    def __call__(
        self,
        result: SolveResult,
        H: TimeQArray,
        parameters: dict | Array,  # noqa ARG002
    ) -> tuple[tuple[Array, Array]]:
        cost = jnp.abs(self.evaluate_controls(result, H, lambda x: x))
        return ((cost, cost < self.target_cost),)

    def __mul__(self, other: float) -> ControlCostArea:
        return ControlCostArea(other * self.cost_multiplier, self.target_cost)


class CustomControlCost(ControlCost):
    cost_fun: callable

    def __call__(
        self,
        result: SolveResult,
        H: TimeQArray,
        parameters: dict | Array,  # noqa ARG002
    ) -> tuple[tuple[Array, Array]]:
        cost = jnp.abs(self.evaluate_controls(result, H, self.cost_fun))
        return ((cost, cost < self.target_cost),)

    def __mul__(self, other: float) -> CustomControlCost:
        return CustomControlCost(
            other * self.cost_multiplier, self.target_cost, self.cost_fun
        )


class CustomCost(Cost):
    cost_multiplier: float
    target_cost: float
    cost_fun: callable

    def __call__(
        self, result: SolveResult, H: TimeQArray, parameters: dict | Array
    ) -> tuple[tuple[Array, Array]]:
        cost = self.cost_fun(result, H, parameters)
        return ((cost, cost < self.target_cost),)

    def __mul__(self, other: float) -> CustomCost:
        return CustomCost(other * self.cost_multiplier, self.target_cost, self.cost_fun)
