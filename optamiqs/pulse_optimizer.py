import equinox as eqx
import jax.tree_util as jtu


class PulseOptimizer(eqx.Module):
    """ Tell the optimizer how to use the parameters we are optimizing to update both
    the Hamiltonian and tsave (for time optimal control).
    """
    H_function: callable
    update_function: callable

    def __init__(self, H_function: callable, update_function: callable):
        self.H_function = jtu.Partial(H_function)
        self.update_function = jtu.Partial(update_function)

    def update(self, drive_params):
        return self.update_function(self.H_function, drive_params)
