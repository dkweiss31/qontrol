# Getting started

qontrol is a quantum optimal control package built on top of [dynamiqs](https://github.com/dynamiqs/dynamiqs). You can define your controls however you would in [dynamiqs](https://github.com/dynamiqs/dynamiqs), specifying only how to update the Hamiltonian at each optimizer step. [dynamiqs](https://github.com/dynamiqs/dynamiqs) also has strong native support for batching, which qontrol can leverage e.g. for randomizing over uncertain parameters.

## Installation

For now we support only installing directly from github
```bash
pip install git+https://github.com/dkweiss31/qontrol
```

Requires Python 3.10+

## Quick example

Optimal control of a Kerr oscillator, with piece-wise constant drives on the I and Q quadratures and optimizing for a `Y` gate

```python
import jax.numpy as jnp
import optax
from dynamiqs import basis, destroy, pwc, dag
import qontrol as qtrl

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


Kerr_model = qtrl.sesolve_model(H_pwc, initial_states, tsave)

parameters = -0.001 * jnp.ones((len(H1s), ntimes - 1))
costs = [qtrl.coherent_infidelity(target_states=target_states), ]

opt_params = qtrl.optimize(
    parameters,
    costs,
    Kerr_model,
    optimizer=optax.adam(learning_rate=0.0001),
    options=qtrl.OptimizerOptions(save_states=False, progress_meter=None),
)
```
We initialize the `sesolve_model` which when called with `parameters` as input runs `sesolve`
and returns that result as well as the updated Hamiltonian. These are in turn passed to 
the cost functions, which tell the optimizer how to update `parameters`.

## Jump in

If this has piqued your interest, please see the example jupyter notebooks that demonstrate different use cases of `qontrol`, including optimizing gates on a qubit to be [robust to frequency variations](examples/qubit) as well as performing [time-optimal control](examples/Kerr_oscillator#time-optimal-control) and [master-equation optimization](examples/Kerr_oscillator#master-equation-optimization). More examples coming soon!

## Citation

If you found this package useful in academic work, please cite

```bibtex
@unpublished{qontrol2024,
  title  = {qontrol: Quantum optimal control based on dynamiqs, diffrax and JAX},
  author = {Daniel K. Weiss},
  year   = {2024},
  url    = {https://github.com/dkweiss31/qontrol}
}
```

Also please consider starring the project on [github](https://github.com/dkweiss31/qontrol/)!