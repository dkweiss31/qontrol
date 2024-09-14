from dynamiqs import Options


class OptimizerOptions(Options):
    """Subclass of dynamiqs Options to allow for various optimizer options.

    Args:
        verbose: If `True`, the optimizer will print out the infidelity at each epoch
            step to track the progress of the optimization.
        target_fidelity: Float that specifies the target fidelity, once hit the
            optimization terminates. Set to 1.0 for the optimization to run through
            all epochs.
        epochs: Number of optimization epochs.
    """

    verbose: bool = True
    target_fidelity: float = 0.995
    epochs: int = 2000

    def __init__(
        self,
        verbose: bool = True,
        target_fidelity: float = 0.9995,
        epochs: int = 1000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.verbose = verbose
        self.target_fidelity = target_fidelity
        self.epochs = epochs
