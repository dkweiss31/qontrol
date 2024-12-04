from __future__ import annotations

import jax
import matplotlib.pyplot as plt
import numpy as np
from dynamiqs.time_array import SummedTimeArray, TimeArray
from IPython.display import clear_output
from jax import Array

from .cost import Cost, SummedCost
from .model import Model
from .options import OptimizerOptions


def _plot_controls_and_loss(  # noqa PLR0915
    parameters: Array | dict,
    costs: Cost,
    model: Model,
    expects: Array | None,
    cost_values_over_epochs: list,
    epoch: int,
    options: OptimizerOptions,
    H_labels: list | None
) -> None:
    # prevents overcrowding of plots in jupyter notebooks
    clear_output(wait=True)
    if expects is not None:
        fig, axs = plt.subplots(ncols=4, figsize=(16, 4))
    else:
        fig, axs = plt.subplots(ncols=3, figsize=(12, 4))
    fig.patch.set_alpha(0.1)
    # first plot the cost values over the range of epochs so far
    ax = axs[0]
    ax.set_facecolor('none')
    epoch_range = np.arange(epoch + 1)
    cost_values_over_epochs = np.asarray(cost_values_over_epochs).T
    if isinstance(costs, SummedCost):
        for _cost, _cost_value in zip(costs.costs, cost_values_over_epochs):
            ax.plot(epoch_range, _cost_value, label=str(_cost))
        ax.plot(
            epoch_range, np.sum(cost_values_over_epochs, axis=0), label='total cost'
        )
    else:
        ax.plot(epoch_range, cost_values_over_epochs[0], label=str(costs))
    ax.set_yscale('log')
    ax.legend(loc='upper right', framealpha=0.0)
    ax.set_xlabel('epochs')
    ax.set_ylabel('cost values')

    # second plot the Hamiltonian prefactors for those terms in the Hamiltonian
    # that have that attribute (e.g. not constant or timecallable)
    ax = axs[1]
    ax.set_facecolor('none')
    H = model.H_function(parameters)
    tsave = model.tsave_function(parameters)
    # use a more discrete time to see steps if pwc, or the full
    # pulse if fitting a spline, etc.
    finer_tsave = np.linspace(0.0, tsave[-1], len(tsave) * 10)

    def evaluate_at_tsave(_H: TimeArray) -> np.ndarray:
        if hasattr(_H, 'prefactor'):
            return jax.vmap(_H.prefactor)(finer_tsave)
        return np.zeros_like(finer_tsave)

    controls = []
    if isinstance(H, SummedTimeArray):
        for _H in H.timearrays:
            controls.append(evaluate_at_tsave(_H))
    else:
        controls.append(evaluate_at_tsave(H))
    for idx, control in enumerate(controls):
        H_label = H_labels[idx] if H_labels else f'$H_{idx}$'
        ax.plot(finer_tsave, np.real(control) / 2 / np.pi, label=H_label)
    ax.legend(loc='lower right', framealpha=0.0)
    ax.set_ylabel('pulse amplitude [GHz]')
    ax.set_xlabel('time [ns]')

    # finally, plot the fft of the controls
    ax = axs[2]
    ax.set_facecolor('none')
    for control_idx, control in enumerate(controls):
        y_fft = np.fft.fft(control)
        n = len(control)
        dt = finer_tsave[1] - finer_tsave[0]
        freqs = np.fft.fftfreq(n, dt)
        ax.plot(freqs[: n // 2], np.abs(y_fft[: n // 2]), label=f'$H_{control_idx}$')
    ax.legend(loc='lower right', framealpha=0.0)
    ax.set_xlabel('frequency [GHz]')
    ax.set_ylabel('fourier amplitude')
    ax.grid(True)

    if expects is not None:
        ax = axs[3]
        ax.set_facecolor('none')
        expects = np.swapaxes(expects, axis1=-2, axis2=-3)
        expect_idxs = np.ndindex(*expects.shape[:-2])
        for state_idx in options.which_states_plot:
            for expect_idx in expect_idxs:
                ax.plot(tsave, np.real(expects[tuple(expect_idx)][state_idx]))
        ax.set_xlabel('time [ns]')
        ax.set_ylabel(f'expectation values for states {options.which_states_plot}')
    plt.tight_layout()
    plt.show()
