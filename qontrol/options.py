from dynamiqs import Options


class GRAPEOptions(Options):
    """Subclass of dynamiqs Options to allow for various GRAPE options.

    Args:
        verbose: If `True`, the optimizer will print out the infidelity at each epoch
            step to track the progress of the optimization.
        target_fidelity: Float that specifies the target fidelity, once hit the
            optimization terminates. Set to 1.0 for the optimization to run through
            all epochs.
        epochs: Number of optimization epochs.
        grape_type: str that specifies if we are doing sesolve or mesolve optimization.
        rng_seed: rng seed for mcsolve.
    """

    verbose: bool
    target_fidelity: float
    epochs: int
    grape_type: str
    rng_seed: int

    def __init__(
        self,
        verbose: bool = True,
        target_fidelity: float = 0.9995,
        epochs: int = 1000,
        grape_type: str = 'sesolve',
        rng_seed: int = 31,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if grape_type == 'sesolve':
            grape_type = 0
        elif grape_type == 'mesolve':
            grape_type = 1
        elif grape_type == 'mcsolve':
            grape_type = 2
        else:
            raise ValueError(
                f"grape_type can be 'sesolve', 'mesolve', or 'mcsolve' but got"
                f'{grape_type}'
            )
        self.verbose = verbose
        self.target_fidelity = target_fidelity
        self.epochs = epochs
        self.grape_type = grape_type
        self.rng_seed = rng_seed
