# Getting started

qontrol is a quantum optimal control package built on top of [dynamiqs](https://github.com/dynamiqs/dynamiqs). You can define your controls however you would in [dynamiqs](https://github.com/dynamiqs/dynamiqs), specifying only how to update the Hamiltonian at each optimizer step. [dynamiqs](https://github.com/dynamiqs/dynamiqs) also has strong native support for batching, which qontrol can leverage e.g. for randomizing over uncertain parameters.

## Installation

For now we support only installing directly from github
```bash
pip install git+https://github.com/dkweiss31/qontrol
```

Requires Python 3.10+


## Documentation

Documentation is available at [https://dkweiss.net/qontrol/](https://dkweiss.net/qontrol/)

## Quick example

Optimal control of a Kerr oscillator, with piece-wise constant drives on the I and Q quadratures and optimizing for a `Y` gate

```python
import jax.numpy as jnp
import optax
from dynamiqs import basis, destroy, pwc, dag
from qontrol import GRAPEOptions, grape, updater, coherent_infidelity

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


params_to_optimize = -0.001 * jnp.ones((len(H1s), ntimes - 1))
costs = [coherent_infidelity(target_states=target_states), ]
H_update = updater(lambda _dp: H_pwc(_dp))
opt_params = grape(
    params_to_optimize,
    costs,
    H_update,
    initial_states,
    tsave=tsave,
    optimizer=optax.adam(learning_rate=0.0001),
    options=GRAPEOptions(save_states=False, progress_meter=None),
)
```
The `updater` function is necessary because we have to tell the optimizer how to update the Hamiltonian once `params_to_optimize` are updated in each round of the optimization. In more complex examples we can also perform time-optimal control where the control times themselves are optimized, see [here](examples/) for example.

## Jump in

If this has piqued your interest, please see the example jupyter notebooks that demonstrate different use cases of `qontrol`, including optimizing qubit pulses to be robust to frequency variations [here](https://github.com/dkweiss31/qontrol/blob/main/docs/examples/qubit.ipynb) as well as performing time-optimal control and master-equation optimization [here](https://github.com/dkweiss31/qontrol/blob/main/docs/examples/Kerr_oscillator.ipynb). Happy optimizing!
