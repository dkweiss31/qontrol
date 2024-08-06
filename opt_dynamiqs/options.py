from dynamiqs import Options


class GRAPEOptions(Options):
    """Subclass of dynamiqs Options to allow for various GRAPE options."""

    target_fidelity: float
    epochs: int
    coherent: bool

    def __init__(
        self,
        target_fidelity: float = 0.9995,
        epochs: int = 1000,
        coherent: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_fidelity = target_fidelity
        self.epochs = epochs
        self.coherent = coherent
