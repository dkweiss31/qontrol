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

    see ADVANCED API for examples

    """
    return Updater(jtu.Partial(update_function))


class Updater(eqx.Module):
    update_function: callable

    def update(self, drive_params: PyTree) -> TimeArray | Array:
        return self.update_function(drive_params)
