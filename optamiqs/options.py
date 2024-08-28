from dynamiqs import Options


class GRAPEOptions(Options):
    """Subclass of dynamiqs Options to allow for various GRAPE options."""

    target_fidelity: float
    epochs: int
    coherent: bool
    grape_type: int
    rng_seed: int

    def __init__(
        self,
        target_fidelity: float = 0.9995,
        epochs: int = 1000,
        coherent: bool = False,
        grape_type: str = "sesolve",
        rng_seed: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if grape_type == "sesolve":
            grape_type = 0
        elif grape_type == "mesolve":
            grape_type = 1
        elif grape_type == "mcsolve":
            grape_type = 2
        else:
            raise ValueError(
                f"grape_type can be 'sesolve', 'mesolve', or 'mcsolve' but got"
                f"{grape_type}"
            )
        self.target_fidelity = target_fidelity
        self.epochs = epochs
        self.coherent = coherent
        self.grape_type = grape_type
        self.rng_seed = rng_seed
