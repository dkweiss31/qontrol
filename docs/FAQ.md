# FAQ

## How can I incorporate drive **constraints** (as opposed to costs)?

Constraints can be incorporated at the level of the Hamiltonian function:

```python
a = dq.destroy(5)
H0 = dq.dag(a) @ dq.dag(a) @ a @ a
H1s = [a + dq.dag(a), 1j * (a - dq.dag(a))]
max_amp = 2.0 * jnp.pi * 0.01
tsave = jnp.linspace(0, 40.0, 31)
parameters = jnp.zeros((len(H1s), len(tsave) - 1))

def H_pwc(drive_params):
    H = H0
    for idx, _H1 in enumerate(H1s):
        clipped_drive = jnp.clip(
            drive_params[idx],
            a_min=-max_amp,
            a_max=max_amp,
        )
        H += dq.pwc(tsave, clipped_drive, _H1)
    return H
```

## How do I initialize random initial pulses?

This can be done by utilizing existing rng functionality from jax. Lets create a random initial guess for the `pwc` control above with values between +1 and -1.
```python
key = jax.random.PRNGKey(31)
parameters = 2.0 * jax.random.uniform(key, (len(H1s), len(tsave) - 1)) - 1.0
```

## How do I monitor custom quantities during optimization?

We provide functionality for the user to plot whatever quantities they'd like to monitor during the course of the optimization. Simply define one or multiple functions with the signature `plotting_function(ax, expects, model, parameters)`, where `ax` is a `matplotlib.pyplot.Axes` object that is updated, `expects` contains the output of `dq.SolveResult.expects`, `model` has type `ql.Model` and `parameters` are the parameters being optimized. Let's imagine you are batching over multiple different frequency values and want a pulse that is robust to these frequency variations. You want to plot the expectation values of each trajectory, and want separate plots for two different initial states. 
```python
n_batch = 21

def plot_states(ax, expects, model, parameters, which=0,):
    ax.set_facecolor('none')
    tsave = model.tsave_function(parameters)
    for batch_idx in range(n_batch):
        ax.plot(tsave, np.real(expects[batch_idx, which, 0]))
    ax.set_xlabel('time [ns]')
    ax.set_ylabel(f'population in $|1\\rangle$ for initial state $|{which}\\rangle$')
    return ax

plotter = ql.custom_plotter(
    [ql.plot_controls,
     functools.partial(plot_states, which=0),
     functools.partial(plot_states, which=1)]
)
```
See [this tutorial](examples/qubit.ipynb) for this sort of functionality in practice. Note there is no limit to the number of panels you can plot: plots will appear in rows with four plots in each row.

## How do I access the saved information?

If a `filepath` is passed to `optimize`, the parameters from each epoch are saved along with the individual values of each cost function and the total cost. This data can be extracted via (assuming the data has been saved in the file 'tmp.h5py')
<!-- skip: next -->
```python
data_dict, param_dict = ql.extract_info_from_h5('tmp.h5py')
```
where `data_dict` contains the optimized parameters as well as cost function info, and `param_dict` contains the options passed to `optimize`.

## You haven't defined my favorite cost function?!

No worries! See the API of [`ql.custom_cost()`][qontrol.custom_cost] as well as [`ql.custom_control_cost()`][qontrol.custom_control_cost] for examples of how to define custom cost functions.
