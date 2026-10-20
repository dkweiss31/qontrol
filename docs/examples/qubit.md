# GRAPE on a qubit with an uncertain frequency

Lets start with the example of GRAPE on a qubit, optimizing the drive envelope to be robust to variations in qubit frequency. This example will illustrate the utility of batching, allowing us to average over many different qubit frequencies to obtain a pulse that achieves a high-fidelity gate and is robust to frequency fluctuations.

This example is available as a Jupyter notebook [here](https://github.com/dkweiss31/qontrol/blob/main/docs/examples/qubit.ipynb).


```python
from functools import partial

import dynamiqs as dq
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array
from jax.random import normal, PRNGKey
from matplotlib.pyplot import Axes

import qontrol as ql
from qontrol.plot import plot_controls
```


```python
time = 30
control_dt = 2.0
ntimes = int(time // control_dt) + 1
optimizer = optax.adam(learning_rate=0.001, b1=0.99, b2=0.99)
tsave_control = jnp.linspace(0, time, ntimes)
tsave_plot = jnp.linspace(0, time, 10 * ntimes)
opt_options = {
    'verbose': False,
    'epochs': 4000,
    'plot': True,
    'plot_period': 200,
    'ftol': 0.0,
}
dq_options = dq.Options(save_states=False, progress_meter=False)
```

Here we initialize the random qubit frequency fluctuations pulled from a normal distribution. Note that the drive is on resonance with the average value of the qubit frequency.


```python
key = PRNGKey(42)
n_batch = 21
random_freqs = 2.0 * jnp.pi * normal(key, shape=(n_batch,)) / 500
H0 = random_freqs[:, None, None] * dq.sigmaz()
```

Here we define what final states the initial states should map to. In this case we want to achieve a Y gate


```python
initial_states = [dq.basis(2, 0), dq.basis(2, 1)]
target_states = [-1j * dq.basis(2, 1), 1j * dq.basis(2, 0)]
```

We next initialize our first guess for the controls and define the function that, given the controls, returns the Hamiltonian.


```python
init_drive_params = {
    'X': -0.001 * jnp.ones(ntimes - 1),
    'Y': -0.001 * jnp.ones(ntimes - 1),
}


def H_pwc(values: dict) -> dq.TimeQArray:
    H = dq.constant(H0)
    H += dq.pwc(tsave_control, values['X'], dq.sigmax())
    H += dq.pwc(tsave_control, values['Y'], dq.sigmay())
    return H


exp_ops = [dq.basis(2, 1) @ dq.dag(dq.basis(2, 1))]
sesolve_model = ql.sesolve_model(H_pwc, initial_states, tsave_plot, exp_ops=exp_ops)
```

In this example we use the coherent definition of the infidelity and penalize drive strengths above 16 MHz


```python
costs = ql.coherent_infidelity(target_states=target_states, target_cost=0.001)
costs += ql.control_norm(2.0 * jnp.pi * 0.016, target_cost=0.1)
```

We modify the default plotter to track both the ground and first excited states individually


```python
def plot_states(
    ax: Axes,
    expects: Array | None,
    model: ql.Model,
    parameters: Array | dict,
    which: int = 0,
) -> Axes:
    ax.set_facecolor('none')
    _tsave = model.tsave_function(parameters)
    for batch_idx in range(n_batch):
        ax.plot(_tsave, np.real(expects[batch_idx, which, 0]))
    ax.set_xlabel('time [ns]')
    ax.set_ylabel(f'population in $|1\\rangle$ for initial state $|{which}\\rangle$')
    return ax


plotter = ql.custom_plotter(
    [plot_controls, partial(plot_states, which=0), partial(plot_states, which=1)]
)
```


```python
opt_params = ql.optimize(
    init_drive_params,
    costs,
    sesolve_model,
    plotter=plotter,
    optimizer=optimizer,
    opt_options=opt_options,
    dq_options=dq_options,
)
```


    
![png](qubit_files/qubit_13_0.png)
    


    target cost reached for all cost functions
    optimization terminated after 1798 epochs
    average epoch time (excluding jit) of 0.00473 s
    max epoch time of 0.00674 s
    min epoch time of 0.00361 s


We see that despite the MHz-level frequency variations of the qubit frequency, the obtained pulse succesfully performs the desired state transfer! Here we plot the population in the $|1\rangle$ state when beginning in the $|0\rangle$ state for the different batch instances
