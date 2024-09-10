# Getting started

qontrol is a quantum optimal control package built on top of [dynamiqs](https://github.com/dynamiqs/dynamiqs). You can define your controls however you would in [dynamiqs](https://github.com/dynamiqs/dynamiqs), specifying only how to update the Hamiltonian and control times at each optimizer step. [dynamiqs](https://github.com/dynamiqs/dynamiqs) also has strong native support for batching, which qontrol can leverage e.g. for randomizing over uncertain parameters.

## Installation

For now we support only installing directly from github
```bash
pip install git+https://github.com/dkweiss31/qontrol
```

Requires Python 3.10+

## Quick example

Optimal control of a Kerr oscillator

```python
import jax.numpy as jnp
import optax
from dynamiqs import basis, destroy, pwc, dag
from qontrol import GRAPEOptions, grape, hamiltonian_time_updater, coherent_infidelity

dim = 5
a = destroy(5)
H0 = -0.5 * 0.2 * dag(a) @ dag(a) @ a @ a
H1s = [a + dag(a), 1j * (a - dag(a))]
initial_states = [basis(dim, 0), basis(dim, 1)]
target_states = [-1j * basis(dim, 1), 1j * basis(dim, 0)]

time = 40
ntimes = int(time // 2.0) + 1
tsave = jnp.linspace(0, time, ntimes)

def H_pwc(drive_params):
    H = H0
    for idx, _H1 in enumerate(H1s):
        H += pwc(tsave, drive_params[idx], _H1)
    return H

ham_time_update = hamiltonian_time_updater(
    H_pwc, lambda _H, _dp: (_H(_dp), tsave)
)
opt_params = grape(
    ham_time_update,
    initial_states=initial_states,
    costs=[coherent_infidelity(target_states=target_states),],
    params_to_optimize=-0.001 * jnp.ones((len(H1s), ntimes - 1)),
    optimizer=optax.adam(learning_rate=0.0001),
    options=GRAPEOptions(save_states=False, progress_meter=None),
)
```
We have to tell the optimizer how to update both the Hamiltonian and the control times (for time-optimal control). Here we don't optimize over the control times, see EXAMPLE for an example where the control times are optimized. 

Time for some more examples!
