from dynamiqs import Options


class OptimizerOptions(Options):
    """Subclass of dynamiqs Options to allow for various optimizer options.

    Args:
        verbose: If `True`, the optimizer will print out the infidelity at each epoch
            step to track the progress of the optimization.
        all_costs: _(bool)_: Whether or not all costs must be below their targets for
            early termination of the optimizer. If False, the optimization terminates
            if only one cost function is below the target (typically infidelity).
        epochs: Number of optimization epochs.
    """

    verbose: bool = True
    all_costs: bool = True
    epochs: int = 2000

    def __init__(
        self, verbose: bool = True, all_costs: bool = True, epochs: int = 2000, **kwargs
    ):
        super().__init__(**kwargs)
        self.verbose = verbose
        self.all_costs = all_costs
        self.epochs = epochs
