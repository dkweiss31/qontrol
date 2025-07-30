from __future__ import annotations

from collections.abc import Callable

import jax
import matplotlib.pyplot as plt
import numpy as np
from dynamiqs.time_qarray import ConstantTimeQArray, SummedTimeQArray, TimeQArray
from IPython.display import clear_output
from jax import Array
from matplotlib.pyplot import Axes

from .cost import Cost, SummedCost
from .model import Model


def plot_costs(
    ax: Axes, costs: Cost, epoch: int, cost_values_over_epochs: list
) -> Axes:
    """Plot the evolution of the cost function values."""
    ax.set_facecolor('none')
    epoch_range = np.arange(epoch + 1)
    cost_values_over_epochs = np.asarray(cost_values_over_epochs).T
    if isinstance(costs, SummedCost):
        for _cost, _cost_value in zip(
            costs.costs, cost_values_over_epochs, strict=True
        ):
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
    return ax


def get_controls(H: TimeQArray, tsave: np.ndarray) -> list[np.ndarray]:
    """Extract the Hamiltonian prefactors at the supplied times."""

    def evaluate_at_tsave(_H: TimeQArray) -> np.ndarray:
        if not isinstance(_H, ConstantTimeQArray):
            return np.asarray(jax.vmap(_H.prefactor)(tsave))
        return np.zeros_like(tsave)

    controls = []
    if isinstance(H, SummedTimeQArray):
        for _H in H.timeqarrays:
            if isinstance(_H, ConstantTimeQArray):
                controls.insert(0, evaluate_at_tsave(_H))
            else:
                controls.append(evaluate_at_tsave(_H))
    else:
        controls.append(evaluate_at_tsave(H))
    return controls


def plot_controls(
    ax: Axes, _expects: Array | None, model: Model, parameters: Array | dict
) -> Axes:
    """Plot the Hamiltonian prefactors, usually corresponding to controls."""
    ax.set_facecolor('none')
    H = model.H_function(parameters)
    tsave = model.tsave_function(parameters)
    controls = get_controls(H, tsave)
    H_labels = [f'$H_{idx}$' for idx in range(len(controls))]
    for idx, control in enumerate(controls):
        ax.plot(tsave, np.real(control), label=H_labels[idx])
    ax.legend(loc='lower right', framealpha=0.0)
    ax.set_ylabel('pulse amplitude')
    ax.set_xlabel('time [ns]')
    return ax


def plot_fft(
    ax: Axes, _expects: Array | None, model: Model, parameters: Array | dict
) -> Axes:
    """Plot the fft of the Hamiltonian controls."""
    ax.set_facecolor('none')
    H = model.H_function(parameters)
    tsave = model.tsave_function(parameters)
    controls = get_controls(H, tsave)
    for control_idx, control in enumerate(controls):
        y_fft = np.fft.fft(control)
        n = len(control)
        dt = tsave[1] - tsave[0]
        freqs = np.fft.fftfreq(n, dt)
        ax.plot(freqs[: n // 2], np.abs(y_fft[: n // 2]), label=f'$H_{control_idx}$')
    ax.legend(loc='lower right', framealpha=0.0)
    ax.set_xlabel('frequency [GHz]')
    ax.set_ylabel('fourier amplitude')
    ax.grid(True)
    return ax


def plot_expects(
    ax: Axes, expects: Array | None, model: Model, parameters: Array | dict
) -> Axes:
    """Plot the expectation values obtained from the time evolution."""
    ax.set_facecolor('none')
    tsave = model.tsave_function(parameters)
    # plot all expectation values by default
    if expects is not None:
        expect_idxs = np.ndindex(*expects.shape[:-1])
        for expect_idx in expect_idxs:
            ax.plot(tsave, np.real(expects[tuple(expect_idx)]))
    ax.set_xlabel('time [ns]')
    ax.set_ylabel('expectation values')
    return ax


def custom_plotter(plotting_functions: list[Callable]) -> Plotter:
    r"""Instantiate a custom Plotter for tracking results during optimization.

    This function returns a Plotter that can be passed to `optimize` to track the
    progress of an optimization run. Note that the cost function values are always
    plotted in the first panel and that there is no limit to the number of plots a user
    can ask for: if more than four, additional plots will appear in a new row of four,
    and so on.

    Args:
        plotting_functions _(list[Callable])_: list of functions that each return a plot
            useful for tracking intermediate results, such as the value of the optimized
            controls, fft of the controls, expectation values, etc. Each function must
            have signature `example_plot_function(ax, expects, model, parameters)` where
            `ax` is the matplotlib.pyplot.Axes instance where the results are plotted,
            `expects` is of type dq.SolveResult.expects (which could be `None`), `model`
            is of type `ql.Model` and `parameters` are the parameters being optimized.
            Of course, some of these arguments may be unused for a particular plot (for
            instance if we are plotting expectation values, we don't need access to
            `parameters`).

    Returns:
        _(Plotter)_: Plotter whose `update_plots` method is repeatedly called during an
            optimization run.

    Examples:
        We plot the controls as well as the expectation values for two different initial
        states
        ```python
        import dynamiqs as dq
        import jax.numpy as jnp
        import numpy as np
        import qontrol as ql
        from functools import partial

        H1s = [dq.sigmax(), dq.sigmay()]
        H1_labels = ['X', 'Y']


        def plot_states(
            ax: Axes,
            expects: Array | None,
            model: ql.Model,
            parameters: Array | dict,
            which=0,
        ) -> Axes:
            ax.set_facecolor('none')
            tsave = model.tsave_function(parameters)
            batch_idxs = np.ndindex(*expects.shape[:-3])
            for batch_idx in batch_idxs:
                ax.plot(tsave, np.real(expects[tuple(batch_idx), which, 0]))
            ax.set_xlabel('time [ns]')
            ax.set_ylabel(
                f'population in $|1\\rangle$ for initial state $|{which}\\rangle$'
            )
            return ax


        def plot_controls(
            ax: Axes, _expects: Array | None, model: ql.Model, parameters: Array | dict
        ) -> Axes:
            ax.set_facecolor('none')
            tsave = model.tsave_function(parameters)
            finer_tsave = jnp.linspace(0.0, tsave[-1], 10 * len(tsave))
            for idx, control in enumerate(parameters):
                H_c = dq.pwc(tsave, control, H1s[idx])
                ax.plot(
                    finer_tsave,
                    np.real(jax.vmap(H_c.prefactor)(finer_tsave)) / 2 / np.pi,
                    label=H1_labels[idx],
                )
            ax.legend(loc='lower right', framealpha=0.0)
            ax.set_ylabel('pulse amplitude [GHz]')
            ax.set_xlabel('time [ns]')
            return ax


        plotter = ql.custom_plotter(
            [
                plot_controls,
                partial(plot_states, which=0),
                partial(plot_states, which=1),
            ]
        )
        ```
        See for example [this tutorial](../examples/qubit).
    """
    return Plotter(plotting_functions)


class Plotter:
    def __init__(self, plotting_functions: list[Callable]):
        self.plotting_functions = plotting_functions
        self.n_plots = len(plotting_functions) + 1

    def update_plots(
        self,
        parameters: Array | dict,
        costs: Cost,
        model: Model,
        expects: Array | None,
        cost_values_over_epochs: list,
        epoch: int,
    ):
        clear_output(wait=True)
        n_col = 4 if self.n_plots >= 4 else self.n_plots
        n_rows = np.ceil(self.n_plots / n_col).astype(int)
        fig, axes = plt.subplots(n_rows, n_col, figsize=(n_col * 4, n_rows * 4))
        if n_rows == 1:
            axes = axes[None]
        fig.patch.set_alpha(0.1)
        axes[0, 0] = plot_costs(axes[0, 0], costs, epoch, cost_values_over_epochs)
        for plot_idx in range(self.n_plots - 1):
            row_idx, col_idx = np.unravel_index(1 + plot_idx, (n_rows, n_col))
            axes[row_idx, col_idx] = self.plotting_functions[plot_idx](
                axes[row_idx, col_idx], expects, model, parameters
            )
        plt.tight_layout()
        plt.show()


class DefaultPlotter(Plotter):
    def __init__(self):
        Plotter.__init__(self, [plot_fft, plot_controls, plot_expects])
