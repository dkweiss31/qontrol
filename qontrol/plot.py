from __future__ import annotations

import jax
import matplotlib.pyplot as plt
import numpy as np
from dynamiqs.time_array import SummedTimeArray, TimeArray, ConstantTimeArray
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
) -> None:
    clear_output(wait=True)
    
    # Calculate the number of rows needed
    ncols = 4
    nrows = (len(options.which_states_plot) + ncols - 1) // ncols + 1  # This ensures rounding up
    
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 4 * nrows))
    fig.patch.set_alpha(0.1)

    # Flatten axs for easy iteration
    axs = axs.flatten()

    # First plot: cost values over the range of epochs
    ax = axs[0]
    ax.set_facecolor('none')
    epoch_range = np.arange(len(cost_values_over_epochs))
    cost_values_over_epochs = np.asarray(cost_values_over_epochs).T
    if isinstance(costs, SummedCost):
        for _cost, _cost_value in zip(costs.costs, cost_values_over_epochs):
            ax.plot(epoch_range, _cost_value, label=str(_cost))
        ax.plot(epoch_range, np.sum(cost_values_over_epochs, axis=0), label='total cost')
    else:
        ax.plot(epoch_range, cost_values_over_epochs[0], label=str(costs))
    ax.set_yscale('log')
    ax.legend(loc='upper right', framealpha=0.0)
    ax.set_xlabel('epochs')
    ax.set_ylabel('cost values')

    # Second plot: Hamiltonian prefactors
    ax = axs[1]
    ax.set_facecolor('none')
    H = model.H_function(parameters)
    tsave = model.tsave_function(parameters)
    finer_tsave = np.linspace(0.0, tsave[-1], len(tsave) * 10)

    def evaluate_at_tsave(_H):
        if hasattr(_H, 'prefactor'):
            return jax.vmap(_H.prefactor)(finer_tsave)

    controls = []
    if isinstance(H, SummedTimeArray):
        for _H in H.timearrays:
            if not isinstance(_H, ConstantTimeArray):
                controls.append(evaluate_at_tsave(_H))
    else:
        controls.append(evaluate_at_tsave(H))
    for idx, control in enumerate(controls):
        ax.plot(finer_tsave, np.real(control) / 2 / np.pi, label=f'$H_{idx+1}$')
    ax.legend(loc='lower right', framealpha=0.0)
    ax.set_ylabel('pulse amplitude [GHz]')
    ax.set_xlabel('time [ns]')

    # Third plot: FFT of the controls
    ax = axs[2]
    ax.set_facecolor('none')
    for control_idx, control in enumerate(controls):
        y_fft = np.fft.fft(control) / len(control)
        n = len(control)
        dt = finer_tsave[1] - finer_tsave[0]
        freqs = np.fft.fftfreq(n, dt)
        ax.plot(freqs[: n // 2], np.abs(y_fft[: n // 2]), label=f'$H_{control_idx}$')
    ax.legend(loc='lower right', framealpha=0.0)
    ax.set_xlabel('frequency [GHz]')
    ax.set_ylabel('fourier amplitude')
    ax.set_xlim(0, options.freq_cutoff)
    ax.grid(True)

    # Plot expectation values
    if expects is not None:
        expects = np.swapaxes(expects, axis1=-2, axis2=-3)
        handles, labels = [], []
        for idx, state_idx in enumerate(options.which_states_plot):
            row_num = (3 + idx) // ncols
            col_num = (3 + idx) % ncols
            ax = axs[row_num * ncols + col_num]
            ax.set_facecolor('none')
            expect_idxs = np.ndindex(*expects.shape[:-2])
            for i, expect_idx in enumerate(expect_idxs):
                line, = ax.plot(tsave, np.real(expects[expect_idx][state_idx]), label=f'state {i}')
                if idx == 0:  # Collect labels and handles only once
                    handles.append(line)
                    labels.append(f'state {i}')
            ax.set_xlabel('time [ns]')
            ax.set_ylabel(f'expectation values')
            ax.set_title(f'initial state is {state_idx}')
            ax.set_yscale('log')
            ax.set_ylim(1e-4,1 )

    # Create a single legend outside the figure
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(labels), framealpha=0.0)
    plt.tight_layout()
    plt.show()