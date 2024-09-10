from __future__ import annotations

import equinox as eqx
import jax.tree_util as jtu
from dynamiqs import TimeArray
from jax import Array
from jaxtyping import PyTree


def hamiltonian_time_updater(
    H_function: callable, update_function: callable
) -> HamiltonianTimeUpdater:
    r"""Hamiltonian and control times updater.

    Specify the Hamiltonian as well as the update function that encodes how to update
    the Hamiltonian and the control times.

    Args:
        H_function _(callable)_: function that accepts the updated parameters as input
            and returns the Hamiltonian
        update_function _(callable)_: function whose signature should be
            (H1: callable, dp: any) -> (H2: dq.TimeArray, tsave: Array)

    see ADVANCED API for examples

    """
    return HamiltonianTimeUpdater(jtu.Partial(H_function), jtu.Partial(update_function))


class HamiltonianTimeUpdater(eqx.Module):
    H_function: callable
    update_function: callable

    def update(self, drive_params: PyTree) -> tuple[TimeArray, Array]:
        return self.update_function(self.H_function, drive_params)
