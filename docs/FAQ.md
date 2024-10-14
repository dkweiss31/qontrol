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

## How do I access the saved information?

If a `filepath` is passed to `optimize`, the parameters from each epoch are saved along with the individual values of each cost function and the total cost. This data can be extracted via (assuming the data has been saved in the file 'tmp.h5py')
<!-- skip: next -->
```python
data_dict, param_dict = ql.extract_info_from_h5('tmp.h5py')
```
where `data_dict` contains the optimized parameters as well as cost function info, and `param_dict` contains the options passed to `optimize`.

## You haven't defined my favorite cost function?!

No worries! See the API of [`ql.custom_cost()`][qontrol.custom_cost] as well as [`ql.custom_control_cost()`][qontrol.custom_control_cost] for examples of how to define custom cost functions.
