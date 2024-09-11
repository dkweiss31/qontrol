from __future__ import annotations

import equinox as eqx
import jax.tree_util as jtu
from dynamiqs import TimeArray
from jax import Array
from jaxtyping import PyTree


def updater(update_function: callable) -> Updater:
    r"""Update system (Hamiltonian or tsave) based on optimized parameters.

    We need to specify how to update objects like the system Hamiltonian `H` and/or the
    control times `tsave` as a function of the parameters we optimize over.

    Args:
        update_function _(callable)_: function that accepts the updated parameters as
            input and returns the system object (Hamiltonian or tsave)

    Returns:
        _(Updater)_: Callable object that updates the Hamiltonian or `tsave`

    Examples:
        In the simplest case, the Hamiltonian function simply takes the optimized
        parameters as input

        ```python
        from jax import Array
        import dynamiqs as dq
        import qontrol as qtrl

        def H_func(values: Array) -> dq.TimeArray:
            # use values to update the Hamiltonian

        ham_time_update = qtrl.updater(lambda values: H_func(values))
        ```
        See for example [this tutorial](../examples/qubit).

        In more complex cases, such as with `timecallable` Hamiltonians, the function
        supplied to `updater` might be more involved
        ```python
        from jax import Array
        import dynamiqs as dq
        import qontrol as qtrl

        def H_func(t: float, drive_params_dict: dict) -> dq.TimeArray:
            # update the Hamiltonian based on drive_params_dict and return
            # the Hamiltonian at time t

        def update_H(drive_params_dict: dict) -> tuple[dq.TimeArray, Array]:
            new_H = jtu.Partial(H_func, drive_params_dict=drive_params_dict)
            return dq.timecallable(new_H)

        ham_time_update = qtrl.updater(update_H)
        ```
        See for example [this tutorial](../examples/Kerr_oscillator#time-optimal-control).

    """  # noqa: E501
    return Updater(jtu.Partial(update_function))


class Updater(eqx.Module):
    update_function: callable

    def update(self, drive_params: PyTree) -> TimeArray | Array:
        return self.update_function(drive_params)
