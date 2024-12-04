from dynamiqs import Options


class OptimizerOptions(Options):
    """Subclass of dynamiqs Options to allow for various optimizer options.

    Args:
        verbose _(bool)_: If `True`, the optimizer will print out the infidelity at
            each epoch step to track the progress of the optimization.
        all_costs _(bool)_: Whether or not all costs must be below their targets for
            early termination of the optimizer. If False, the optimization terminates
            if only one cost function is below the target (typically infidelity).
        epochs _(int)_: Number of optimization epochs.
        plot _(bool)_: Whether to plot the results during the optimization (for the
            epochs where results are plotted, necessarily suffer a time penalty).
        plot_period _(int)_: If plot is True, plot every plot_period.
        which_states_plot _(tuple)_: Which states to plot if expectation values are
            passed to the model. Defaults to (0, ), so just plot expectation values for
            the zero indexed batch state
        xtol _(float)_: Defaults to 1e-8, terminate the optimization if the parameters
            are not being updated
        ftol _(float)_: Defaults to 1e-8, terminate the optimization if the cost
            function is not changing above this level
        gtol _(float)_: Defaults to 1e-8, terminate the optimization if the norm of the
            gradient falls below this level
    """

    verbose: bool
    all_costs: bool
    epochs: int
    plot: bool
    plot_period: int
    which_states_plot: tuple
    xtol: float
    ftol: float
    gtol: float
    freq_cutoff: float

    def __init__(
        self,
        verbose: bool = True,
        all_costs: bool = True,
        epochs: int = 2000,
        plot: bool = True,
        plot_period: int = 30,
        which_states_plot: tuple = (0,),
        xtol: float = 1e-8,
        ftol: float = 1e-8,
        gtol: float = 1e-8,
        freq_cutoff: float = 10.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.verbose = verbose
        self.all_costs = all_costs
        self.epochs = epochs
        self.plot = plot
        self.plot_period = plot_period
        self.which_states_plot = which_states_plot
        self.xtol = xtol
        self.ftol = ftol
        self.gtol = gtol
        self.freq_cutoff = freq_cutoff
