import jax
import jax.random as jr
import equinox as eqx
import equinox.nn as enn

from spowl.spowl import layers, init


class CostModel(eqx.Module):
    _encoder: eqx.Module
    _dynamics: eqx.Module
    _cost: eqx.Module
    
    def __init__(
        self,
        seed: int,
        cost_mlp_act: str,
        cm_dropout: float,
        cm_enc_dim: int,
        obs_dim: int,
        action_dim: int,
        cm_enc_layers: int,
        cm_enc_act: str,
        cm_simnorm_dim: int,
        cm_state_dim: int,
        cm_dim: int,
        cm_layers: int,
        cm_dyn_act: str,
    ):
        key = jr.key(seed)
        if cost_mlp_act == 'tanh':
            act = lambda: enn.Lambda(jax.nn.tanh)
        elif cost_mlp_act == 'mish':
            act = lambda: enn.Lambda(jax.nn.mish)
        elif cost_mlp_act == 'relu':
            act = lambda: enn.Lambda(jax.nn.relu)
        
        layer_fn = lambda in_dim, out_dim, key, a: layers.NormedLinear(
            in_dim, out_dim, a, dropout=cm_dropout, key=key
        )
        
        # Create encoder
        dims = [obs_dim] + [cm_enc_dim] * (cm_enc_layers-1)
        encoder = []
        for i in range(cm_enc_layers-1):
            key, sub_key = jr.split(key)
            encoder.append(layer_fn(dims[i], dims[i+1], sub_key, act()))
        key, sub_key = jr.split(key)
        if cm_enc_act == 'simnorm':
            enc_act = layers.SimNorm(cm_simnorm_dim)
        elif cm_enc_act == 'identity':
            enc_act = enn.Lambda(lambda x: x)
        elif cm_enc_act == 'sigmoid':
            enc_act = enn.Lambda(jax.nn.sigmoid)
        elif cm_enc_act == 'mish':
            enc_act = enn.Lambda(jax.nn.mish)
        elif cm_enc_act == 'relu':
            enc_act = enn.Lambda(jax.nn.relu)
        encoder.append(layer_fn(dims[-1], cm_state_dim, sub_key, enc_act))
        self._encoder = enn.Sequential(encoder)
        
        # Create dynamics head
        dims = [cm_state_dim + action_dim] + [cm_dim] * (cm_layers-1)
        dynamics = []
        for i in range(cm_layers-1):
            key, sub_key = jr.split(key)
            dynamics.append(layer_fn(dims[i], dims[i+1], sub_key, act()))
        if cm_dyn_act == 'simnorm':
            dyn_act = layers.SimNorm(cm_simnorm_dim)
        elif cm_dyn_act == 'identity':
            dyn_act = enn.Lambda(lambda x: x)
        key, sub_key = jr.split(key)
        dynamics.append(layer_fn(dims[-1], cm_state_dim, sub_key, dyn_act))
        self._dynamics = enn.Sequential(dynamics)
        
        # Create cost head
        dims = [cm_state_dim + action_dim] + [cm_dim] * (cm_layers-1)
        cost = []
        for i in range(cm_layers-1):
            key, sub_key = jr.split(key)
            cost.append(layer_fn(dims[i], dims[i+1], sub_key, act()))
        key, sub_key = jr.split(key)
        cost.append(enn.Linear(dims[-1], 1, key=sub_key))
        cost.append(enn.Lambda(jax.nn.softplus))        
        self._cost = enn.Sequential(cost)
    
    def __call__(self, obs, action, key):
        key1, key2 = jax.random.split(key)
        s = self.encode(obs, key=key1)
        return self.cost(s, action, key2)
    
    def encode(self, obs, key):
        return self._encoder(obs, key=key)
    
    def imagine(self, obs, actions, key):
        key1, key2 = jax.random.split(key)
        state = self.encode(obs, key1)
        states = self.imagine_from_state(state, actions, key2)
        return states
    
    def imagine_from_state(self, state, actions, key):
        def step(state, data):
            action, key = data
            next_s = self.imagine_step(state, action, key)
            return next_s, next_s
        
        _, next_states = jax.lax.scan(
            step, state, (actions, jax.random.split(key, actions.shape[0]))
        )
        states = jax.numpy.concatenate([state[None], next_states], 0)
        return states
    
    def imagine_cost(self, state, actions, key):
        key1, key2 = jax.random.split(key)
        states = self.imagine_from_state(state, actions, key1)[:-1]
        cost = jax.vmap(self.cost)(states, actions, jax.random.split(key2, states.shape[0]))
        return cost
    
    def imagine_step(self, state, action, key):
        sa = jax.numpy.concatenate([state, action], axis=-1)
        next_s = self._dynamics(sa, key=key)
        return next_s
    
    def cost(self, state, action, key):
        sa = jax.numpy.concatenate([state, action], axis=-1)
        out = self._cost(sa, key=key)
        return out
    
    def rollout(self, observations, actions, dones, key):
        key1, key2, key3 = jax.random.split(key, 3)
        states = jax.vmap(self.encode)(observations, jax.random.split(key1, observations.shape[0]))
        init_state = states[0]
        next_states = jax.lax.stop_gradient(states[1:])
        def step(state, data):
            n_state, action, done, key = data
            next_s = self.imagine_step(state, action, key)
            next_s = jax.numpy.where(done, n_state, next_s)
            return next_s, next_s
        
        _, pred_states = jax.lax.scan(
            step, init_state, (next_states, actions, dones, jax.random.split(key2, actions.shape[0]))
        )
        state_error = jax.numpy.square(next_states - pred_states).mean(-1, keepdims=True)
        states = jax.numpy.concatenate([init_state[None], pred_states[:-1]])
        # predict costs on rollout
        pred_cost = jax.vmap(self.cost)(states, actions, jax.random.split(key3, states.shape[0]))
        return state_error, pred_cost
    
    
def make_cost_model(seed, use_custom_init, last_zero, *args) -> CostModel:
    key = jr.key(seed)
    model = CostModel(seed, *args)
    if use_custom_init:
        model = init.init_linear(model, key)
    if last_zero:
        model = eqx.tree_at(lambda m: m._cost.layers[-2].weight, model, jax.numpy.zeros_like(model._cost.layers[-2].weight))
    return model
