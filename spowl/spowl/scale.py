import jax
import jax.numpy as jnp
import equinox as eqx


class RunningScale(eqx.Module):
    tau: float
    _value: jax.Array
    _percentiles: jax.Array

    def __init__(self, tau: float):
        self.tau = tau
        self._value = jnp.ones(1)
        self._percentiles = jnp.array([5, 95])

    @property
    def value(self):
        return self._value

    def _percentile(self, x): 
        x_shape = x.shape
        x = x.reshape((x.shape[0], -1))
        in_sorted = jnp.sort(x, axis=0)
        positions = self._percentiles * (x.shape[0]-1) / 100
        floored = jnp.floor(positions)
        ceiled = floored + 1
        ceiled = jnp.where(ceiled > x.shape[0] - 1, x.shape[0] - 1, ceiled)
        weight_ceiled = positions - floored
        weight_floored = 1.0 - weight_ceiled
        d0 = in_sorted[floored.astype(jnp.int32), :] * weight_floored[:, None]
        d1 = in_sorted[ceiled.astype(jnp.int32), :] * weight_ceiled[:, None]
        return (d0+d1).reshape((-1, *x_shape[1:]))

    def update(self, x):
        percentiles = self._percentile(x)
        value = jnp.clip(percentiles[1] - percentiles[0], min=1.)
        return self._value + self.tau * (value - self._value)

    def __call__(self, x, update=False):
        if update:
            self.update(x)
        return x * (1/self.value)

    def __repr__(self):
        return f'RunningScale(S: {self.value})'
