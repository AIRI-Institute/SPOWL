import jax
import jax.random as jr
import jax.numpy as jnp
import jax.tree_util as jtu
import equinox as eqx
import optax

from spowl.spowl import layers, math, init


@eqx.filter_vmap(in_axes=(eqx.if_array(0), None, 0))
def evaluate_ensemble(model, x, key):
    return model(x, key=key)


def inference(model, mode=True):
    model = eqx.nn.inference_mode(model, mode)
    model = eqx.tree_at(
        lambda tree: (tree._target_Qs, tree._target_Cs), model,
        (eqx.nn.inference_mode(model._target_Qs, True), eqx.nn.inference_mode(model._target_Cs, True))
    )
    return model


class WorldModel(eqx.Module):
    _encoder: eqx.Module
    _decoder: eqx.Module = None
    _dynamics: eqx.Module
    _reward: eqx.Module
    _cost: eqx.Module = None
    _pi: eqx.Module
    _Qs: eqx.Module
    _Cs: eqx.Module
    _target_Qs: eqx.Module
    _target_Cs: eqx.Module
    log_std_min: float
    log_std_dif: float
    lagrange_multiplier: float
    penalty_multiplier: float
    num_q: int
    num_c: int
    vmin: float
    vmax: float
    num_bins: int
    c_random_agg: int
    use_cost: bool
    use_decoder: bool

    def __init__(self,
            seed: int,
            obs_dim: int,
            num_enc_layers: int,
            enc_dim: int,
            latent_dim: int,
            simnorm_dim: int,
            action_dim: int,
            mlp_dim: int,
            num_bins: int,
            num_c: int,
            cm_dropout: float,
            num_q: int,
            dropout: float,
            log_std_min: float,
            log_std_max: float,
            lagrange_multiplier_init: float,
            penalty_multiplier_init: float,
            vmin: float,
            vmax: float,
            c_random_agg: int,
            use_cost: bool,
            use_decoder: bool,
    ):
        k1, k2 = jr.split(jr.key(seed))
        keys, ikeys  = jr.split(k1, 8), jr.split(k2, 8)
        
        # Initialize heads
        self._encoder = init.init_linear(
            layers.mlp(keys[0], obs_dim, max(num_enc_layers-1, 1)*[enc_dim], latent_dim, act=layers.SimNorm(simnorm_dim)),
            ikeys[0]
        )
        
        self.use_decoder = use_decoder
        if use_decoder:
            self._decoder = init.init_linear(
                layers.mlp(keys[7], latent_dim, max(num_enc_layers-1, 1)*[enc_dim], obs_dim), ikeys[7]
            )
        self._dynamics = init.init_linear(
            layers.mlp(keys[1], latent_dim + action_dim, 2*[mlp_dim], latent_dim, act=layers.SimNorm(simnorm_dim)),
            ikeys[1]
        )
        self._reward = init.init_linear(
            layers.mlp(keys[2], latent_dim + action_dim, 2*[mlp_dim], max(num_bins, 1)),
            ikeys[2]
        )
        
        self.use_cost = use_cost
        if use_cost:
            self._cost = init.init_linear(
                layers.ens(jr.split(keys[3], num_c), latent_dim + action_dim, 2*[mlp_dim], max(num_bins, 1), None, cm_dropout),
                ikeys[3]
            )
        
        self._pi = init.init_linear(
            layers.mlp(keys[4], latent_dim, 2*[mlp_dim], 2*action_dim),
            ikeys[4]
        )
        self._Qs = init.init_linear(
            layers.ens(jr.split(keys[5], num_q), latent_dim + action_dim, 2*[mlp_dim], max(num_bins, 1), None, dropout),
            ikeys[5]
        )
        self._Cs = init.init_linear(
            layers.ens(jr.split(keys[6], num_c), latent_dim + action_dim, 2*[mlp_dim], max(num_bins, 1), None, dropout),
            ikeys[6]
        )
        
        # Create targets (use same keys)
        self._target_Qs = init.init_linear(
            layers.ens(jr.split(keys[5], num_q), latent_dim + action_dim, 2*[mlp_dim], max(num_bins, 1), None, dropout),
            ikeys[5]
        )
        self._target_Cs = init.init_linear(
            layers.ens(jr.split(keys[6], num_c), latent_dim + action_dim, 2*[mlp_dim], max(num_bins, 1), None, dropout),
            ikeys[6]
        )
        
        # Update last layer weights
        last_zero = lambda x: eqx.tree_at(lambda m: m.layers[-1].weight, x, jnp.zeros_like(x.layers[-1].weight))
        self._reward = last_zero(self._reward)
        self._Qs = last_zero(self._Qs)
        self._target_Qs = last_zero(self._target_Qs)
        if use_cost:
            self._cost = last_zero(self._cost)
        self._Cs = last_zero(self._Cs)
        self._target_Cs = last_zero(self._target_Cs)

        self.log_std_min = log_std_min
        self.log_std_dif = log_std_max - log_std_min
        self.lagrange_multiplier = jnp.float32(lagrange_multiplier_init)
        self.penalty_multiplier = jnp.float32(penalty_multiplier_init)
        
        self.num_c = num_c
        self.num_q = num_q
        self.vmin = vmin
        self.vmax = vmax
        self.num_bins = num_bins
        self.c_random_agg = c_random_agg

    @property
    def total_params(self):
        params = jax.tree.flatten(eqx.filter(self, eqx.is_inexact_array))[0]
        return sum(jax.numpy.size(p) for p in params)

    def encode(self, obs, key):
        return self._encoder(obs, key=key)
    
    def decode(self, latent, key):
        return self._decoder(latent, key=key)

    def next(self, z, a):
        z = jnp.concat([z, a], axis=-1)
        return self._dynamics(z)
    
    def reward(self, z, a):
        z = jnp.concat([z, a], axis=-1)
        return self._reward(z)

    def cost(self, key, z, a):
        z = jnp.concat([z, a], axis=-1)
        return evaluate_ensemble(self._cost, z, jr.split(key, self.num_c))

    @eqx.filter_jit
    def pi(self, key, z):
        # Gaussian policy prior
        mu, log_std = jnp.split(self._pi(z), 2, axis=-1)
        log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
        eps = jr.normal(key, mu.shape)
        log_pi = math.gaussian_logprob(eps, log_std, size=None)
        pi = mu + eps * jnp.exp(log_std)
        mu, pi, log_pi = math.squash(mu, pi, log_pi)
        return mu, pi, log_pi, log_std

    def Q(self, key, z, a, return_type='all', target=False):
        if return_type not in {'min', 'avg', 'all', 'std'}: raise ValueError
        
        z = jnp.concat([z, a], axis=-1)
        e_key, i_key = jr.split(key)
        out = evaluate_ensemble(self._target_Qs if target else self._Qs, z, jr.split(e_key, self.num_q))

        if return_type == 'all':
            return out
        
        # TODO: check indices for batch
        qidx = jr.choice(i_key, self.num_q, (2,), replace=False)
        Q = math.two_hot_inv(out[qidx], self.num_bins, self.vmin, self.vmax)
        
        if return_type == 'min':
            return Q.min(0)
        elif return_type == 'std':
            return Q.std(0)
        else:
            return Q.mean(0)
    
    def C(self, key, z, a, return_type='all', target=False):
        if return_type not in {'min', 'avg', 'all', 'max', 'std'}: raise ValueError
        
        z = jnp.concat([z, a], axis=-1)
        e_key, i_key = jr.split(key)
        out = evaluate_ensemble(self._target_Cs if target else self._Cs, z, jr.split(e_key, self.num_c))

        if return_type == 'all':
            return out

        cidx = jr.choice(i_key, self.num_c, (self.c_random_agg,), replace=False)
        C = math.two_hot_inv(out[cidx], self.num_bins, self.vmin, self.vmax)
        
        if return_type == 'min':
            return C.min(0)
        elif return_type == 'max':
            return C.max(0)
        elif return_type == 'std':
            return C.std(0)
        else:
            return C.mean(0)


def make_world_model(seed, obs_dim, act_dim, wm_config, opt_config):
    model = WorldModel(
        seed,
        obs_dim,
        wm_config.num_enc_layers,
        wm_config.enc_dim,
        wm_config.latent_dim,
        wm_config.simnorm_dim,
        act_dim,
        wm_config.mlp_dim,
        wm_config.num_bins,
        wm_config.num_c,
        wm_config.cm_dropout,
        wm_config.num_q,
        wm_config.dropout,
        wm_config.log_std_min,
        wm_config.log_std_max,
        wm_config.lagrange_multiplier_init,
        wm_config.penalty_multiplier_init,
        wm_config.vmin,
        wm_config.vmax,
        wm_config.c_random_agg,
        wm_config.use_cost,
        wm_config.use_decoder,
    )

    no_grads = eqx.tree_at(
        lambda tree: eqx.tree_flatten_one_level(tree)[0], model,
        replace_fn=lambda _: False
    )
    is_inexact = lambda y: jtu.tree_map(lambda x: eqx.is_inexact_array(x), y)
    
    # Initialize WM optimizer
    if model.use_cost:
        filter_spec = eqx.tree_at(
            lambda tree: (
                tree._encoder, tree._dynamics, tree._reward, tree._cost, tree._Qs, tree._Cs
            ), no_grads, replace=(
                is_inexact(model._encoder),
                is_inexact(model._dynamics),
                is_inexact(model._reward),
                is_inexact(model._cost),
                is_inexact(model._Qs),
                is_inexact(model._Cs),
            )
        )
    else:
        filter_spec = eqx.tree_at(
            lambda tree: (
                tree._encoder, tree._dynamics, tree._reward, tree._Qs, tree._Cs
            ), no_grads, replace=(
                is_inexact(model._encoder),
                is_inexact(model._dynamics),
                is_inexact(model._reward),
                is_inexact(model._Qs),
                is_inexact(model._Cs),
            )
        )
        
    if model.use_decoder:
        filter_spec = eqx.tree_at(
            lambda tree: (tree._decoder,), filter_spec, replace=(is_inexact(model._decoder),)
        )
        
    labels = eqx.tree_at(
        lambda tree: eqx.tree_flatten_one_level(tree)[0], model,
        replace_fn=lambda _: 'all'
    )
    labels = eqx.tree_at(lambda tree: tree._encoder, labels, replace='enc')
    separate_transform = optax.multi_transform({
        'enc': optax.adam(learning_rate=opt_config.lr*opt_config.enc_lr_scale),
        'all': optax.adam(learning_rate=opt_config.lr),
    }, labels)
    optim = optax.chain(
        optax.clip_by_global_norm(opt_config.grad_clip_norm),
        separate_transform,
    )
    wm_state = eqx.filter(model, filter_spec) 
    opt_state = optim.init(wm_state)
    
    # Initialize Pi optimizer
    pi_filter_spec = eqx.tree_at(
        lambda tree: tree._pi, no_grads, replace=is_inexact(model._pi)
    )
    pi_optim = optax.chain(
        optax.clip_by_global_norm(opt_config.grad_clip_norm),
        optax.adam(learning_rate=opt_config.lr, eps=1e-5),
    )
    pi_state = eqx.filter(model, pi_filter_spec)
    pi_opt_state = pi_optim.init(pi_state)
    model = inference(model, True)
    return model, (filter_spec, optim, opt_state), (pi_filter_spec, pi_optim, pi_opt_state) 