from dynamiqs import Options


class OptimizerOptions(Options):
    """Subclass of dynamiqs Options to allow for various optimizer options.

    Args:
        verbose _(bool)_: If `True`, the optimizer will print out the infidelity at each epoch
            step to track the progress of the optimization.
        all_costs _(bool)_: Whether or not all costs must be below their targets for
            early termination of the optimizer. If False, the optimization terminates
            if only one cost function is below the target (typically infidelity).
        epochs _(int)_: Number of optimization epochs.
        plot _(bool)_: Whether to plot the results during the optimization (for the epochs
            where results are plotted, necessarily suffer a time penalty).
        plot_period _(int)_: If plot is True, plot every plot_period.

    """

    verbose: bool
    all_costs: bool
    epochs: int
    plot: bool
    plot_period: int

    def __init__(
        self,
        verbose: bool = True,
        all_costs: bool = True,
        epochs: int = 2000,
        plot: bool = True,
        plot_period: int = 30,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.verbose = verbose
        self.all_costs = all_costs
        self.epochs = epochs
        self.plot = plot
        self.plot_period = plot_period
