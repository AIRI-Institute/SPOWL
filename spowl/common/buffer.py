import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
from dataclasses import dataclass

from spowl.common.trajectory import TrajectoryData


@jax.tree_util.register_dataclass
@dataclass
class Transitions:
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    cost: jax.Array
    term: jax.Array
    trunc: jax.Array


class Buffer():
    def __init__(
        self,
        observation_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        max_length: int,
        num_envs: int,
    ):
        self._state = Transitions(
            jnp.zeros((num_envs, max_length, *observation_shape), dtype=jnp.float32),
            jnp.zeros((num_envs, max_length, *action_shape), dtype=jnp.float32),
            jnp.zeros((num_envs, max_length, 1), dtype=jnp.float32),
            jnp.zeros((num_envs, max_length, 1), dtype=jnp.float32),
            jnp.zeros((num_envs, max_length, 1), dtype=jnp.bool),
            jnp.zeros((num_envs, max_length, 1), dtype=jnp.bool),
        )
        self.max_length = max_length
        self.step_id = 0
        self.filled_level = 0
        self.num_envs = num_envs

    @property
    def volume(self):
        return self.filled_level * self.num_envs
    
    @property
    def is_full(self):
        return self.filled_level == self.max_length

    def add(self, data: TrajectoryData, start: int, end: int):
        seq_len = end - start 
        indexes = jnp.remainder(self.step_id + jnp.arange(seq_len), self.max_length)
        data = Transitions(
            jnp.array(data.observation[:, start: end]),
            jnp.array(data.action[:, start: end]),
            jnp.array(data.reward[:, start: end]),
            jnp.array(data.cost[:, start: end]),
            jnp.array(data.terminated[:, start: end]),
            jnp.array(data.truncated[:, start: end]),
        )
        self._state = jt.map(
                lambda x, y: x.at[:, indexes].set(y), self._state, data
            )
        self.filled_level = min(self.filled_level + seq_len, self.max_length)
        self.step_id = (self.step_id + seq_len) % self.max_length
    
    def get_metrics(self):
        if self.is_full:
            cost = self._state.cost
        else:
            cost = self._state.cost[:, :self.step_id]
        metrics = {
            'volume': self.volume,
            'cost_mean': float(cost.mean()),
            'cost_max': float(cost.max()),
        }
        return metrics
    

def sample_from_buffer(
    key: jr.PRNGKey,
    batch_size: int,
    horizon: int,
    state: Transitions,
    is_full: bool,
    step_id: int
):
    print('Compiling sample function ...')
    k1, k2 = jr.split(key)
    if is_full:
        idxs1 = jr.randint(k1, (batch_size,), minval=0, maxval=state.obs.shape[1] - horizon) + step_id
    else:
        idxs1 = jr.randint(k1, (batch_size,), minval=0, maxval=step_id - horizon)
    idxs0 = jr.randint(k2, (batch_size,), minval=0, maxval=state.obs.shape[0])

    idxs0 = jnp.repeat(idxs0[None], horizon + 1, axis=0)
    idxs1 = jnp.repeat(idxs1[None], horizon + 1, axis=0)

    idxs1 = idxs1 + jnp.arange(horizon + 1).reshape((-1, 1))
    idxs1 = jnp.remainder(idxs1, state.obs.shape[1])
    sample = jt.map(lambda x: x[idxs0, idxs1], state)

    obs = sample.obs
    action = sample.action[1:]
    reward = sample.reward[1:]
    cost = sample.cost[1:]
    term = sample.term[1:]
    done = jnp.logical_or(sample.term, sample.trunc)
    return obs, action, reward, cost, term, done