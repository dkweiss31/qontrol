import equinox as eqx
import jax.tree_util as jtu


__all__ = ["PulseOptimizer"]


class PulseOptimizer(eqx.Module):
    """ Tell the optimizer how to use the parameters we are optimizing to update the Hamiltonian.
    """
    H_function: callable
    update_H_function: callable

    def __init__(self, H_function: callable, update_H_function: callable):
        self.H_function = jtu.Partial(H_function)
        self.update_H_function = jtu.Partial(update_H_function)

    def update(self, drive_params):
        return self.update_H_function(self.H_function, drive_params)
