import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

import equinox as eqx

from spowl.spowl.cost_model import CostModel
from spowl.spowl.world_model import WorldModel
from spowl.spowl import math


def _estimate_value(
    model: WorldModel,
    key: jr.PRNGKey,
    z: jax.Array,
    actions: jax.Array,
    gamma: float,
):
    def one_step(carry, act):
        z, G, discount = carry
        reward = jnp.squeeze(math.two_hot_inv(
            jax.vmap(model.reward)(z, act), model.num_bins, model.vmin, model.vmax
        ))
        z = jax.vmap(model.next)(z, act)
        G = G + discount * reward
        discount *= gamma
        return (z, G, discount), None
    
    G, discount = jnp.zeros(z.shape[0]), jnp.array(1)
    init = (z, G, discount)
    (z, G, discount), _ = jax.lax.scan(one_step, init, actions)
    pi_key, q_key = jr.split(key)
    last_action = jax.vmap(model.pi)(jr.split(pi_key, z.shape[0]), z)[1]
    q_func = jax.vmap(lambda k, z, a: model.Q(k, z, a, return_type='avg'))
    q_val = jnp.squeeze(q_func(jr.split(q_key, z.shape[0]), z, last_action))
    return G + discount * q_val


def _constrained_estimate_value(
    model: WorldModel,
    cost_model: CostModel,
    key: jr.PRNGKey,
    latent: dict[str, jax.Array],
    actions: jax.Array,
    gamma: float,
    cost_gamma: float,
    use_cm: bool,
    use_cost: bool,
):
    num_trajs = actions.shape[1]
    def one_step(carry, data):
        z, G, discount, C_val, C, cost_discount = carry
        act, key = data
        reward = jnp.squeeze(math.two_hot_inv(
            jax.vmap(model.reward)(z, act), model.num_bins, model.vmin, model.vmax
        ))
        if use_cost and model.use_cost:
            cost = jnp.squeeze(math.two_hot_inv(
                jax.vmap(model.cost)(jr.split(key, z.shape[0]), z, act), model.num_bins, model.vmin, model.vmax
            ), axis=2)
        else:
            cost = jnp.zeros((num_trajs, 1))
        z = jax.vmap(model.next)(z, act)
        C += cost.max(1)
        C_val += cost_discount * cost.mean(1)
        G += discount * reward
        
        discount *= gamma
        cost_discount *= cost_gamma
        return (z, G, discount, C_val, C, cost_discount), None
    
    plan_key, pi_key, q_key, c_key, cm_key = jr.split(key, 5)
    G, discount = jnp.zeros(num_trajs), jnp.array(1)
    C_val, C, cost_discount = jnp.zeros(num_trajs), jnp.zeros(num_trajs), jnp.array(1)
    init = (latent['reward'], G, discount, C_val, C, cost_discount)
    (z, G, discount, C_val, C, cost_discount), _ = jax.lax.scan(one_step, init, (actions, jr.split(plan_key, actions.shape[0]))) 
    last_action = jax.vmap(model.pi)(jr.split(pi_key, num_trajs), z)[1]

    q_func = jax.vmap(lambda k, z, a: model.Q(k, z, a, return_type='avg'))
    q_val = jnp.squeeze(q_func(jr.split(q_key, num_trajs), z, last_action))
    value = G + discount * q_val
    
    c_func = jax.vmap(lambda k, z, a: model.C(k, z, a, return_type='avg'))
    c_val = jnp.squeeze(c_func(jr.split(c_key, num_trajs), z, last_action))
    cost_value = C_val + cost_discount * c_val
    
    if use_cm:
        C = jax.vmap(cost_model.imagine_cost, in_axes=(0, 1, 0), out_axes=1)(
            latent['cost'], actions, jr.split(cm_key, num_trajs)
        )
        cost_discount = jnp.power(cost_gamma, jnp.arange(C.shape[0]))
        cost_value = jnp.sum(cost_discount[:, None, None] * C, (0, 2))
        cost_value = cost_value + (cost_gamma ** C.shape[0]) * c_val
        C = jnp.sum(C, (0, 2))
    return value, cost_value, C


def get_action_stats(
    key: jr.PRNGKey,
    model: WorldModel,
    state: jax.Array,
    action: jax.Array,
    n_samples: int,
    ci: float
):
    key, sub_key = jr.split(key)
    qs =jnp.squeeze(math.two_hot_inv(
        model.Q(sub_key, state, action), model.num_bins, model.vmin, model.vmax
    ), axis=1)
    key, sub_key = jr.split(key)
    cs =jnp.squeeze(math.two_hot_inv(
        model.C(sub_key, state, action), model.num_bins, model.vmin, model.vmax
    ), axis=1)

    if cs.shape[0] != 1:
        key, sub_key = jr.split(key)
        cs_l, cs_h = math.bootstrap_mean_ci(sub_key, cs, n_samples, ci)
    else:
        cs_l, cs_h = cs[0], cs[0]
        
    if qs.shape[0] != 1:
        key, sub_key = jr.split(key)
        qs_l, qs_h = math.bootstrap_mean_ci(sub_key, qs, n_samples, ci)
    else:
        qs_l, qs_h = qs[0], qs[0]
    return (cs_l, cs_h), (qs_l, qs_h)


def plan(
    model: WorldModel,
    cost_model: CostModel,
    key: jr.PRNGKey,
    latent: dict[str, jax.Array],
    prev_mean: jax.Array,
    discount: float,
    cost_discount: float,
    eval_mode: bool,
    horizon: int,
    num_pi_trajs: int,
    num_samples: int,
    min_std: float,
    max_std: float,
    plan_method: str,
    num_elites: int,
    temperature: float,
    momentum: float,
    iterations: int,
    threshold: float,
    use_local_cost: bool,
    bootstrap_samples: int,
    bootstrap_ci: int,
    use_ci_thresholds: bool,
    restrict_elites: bool,
    restrict_action: bool,
    use_cm: bool,
    use_cost: bool,
):	
    action_dim = prev_mean.shape[-1]
    key, sub_key = jr.split(key)
    current_z = latent['reward']
    if eval_mode:
        pi_a = model.pi(sub_key, current_z) [0]
    else:
        pi_a = model.pi(sub_key, current_z)[1]
        
    key, sub_key = jr.split(key)
    (cs_l, cs_h), (qs_l, qs_h) = get_action_stats(sub_key, model, latent['reward'], pi_a, bootstrap_samples, bootstrap_ci)
    metrics = {
        'reward_value_CI_low': qs_l,
        'reward_value_CI_high': qs_h,
        'cost_value_CI_low': cs_l,
        'cost_value_CI_high': cs_h,
    }
    
    # Sample policy trajectories
    latent = jtu.tree_map(lambda x: x.reshape((1, -1)), latent)
    def pi_step(_z, key):
        acts = jax.vmap(model.pi)(jr.split(key, _z.shape[0]), _z)[1]
        _z = jax.vmap(model.next)(_z, acts)
        return _z, acts
    
    if num_pi_trajs > 0:
        _z = jnp.repeat(latent['reward'], num_pi_trajs, 0)
        key, sub_key = jr.split(key)
        _, pi_actions = jax.lax.scan(pi_step, _z, jr.split(sub_key, horizon))

    # Initialize state and parameters
    latent = jtu.tree_map(lambda x: jnp.repeat(x, num_samples, 0), latent)
    mean = jnp.zeros((horizon, action_dim))
    std = jnp.full((horizon, action_dim), max_std)
    mean = mean.at[:-1].set(prev_mean[1:])

    # Iterate MPPI
    def iteration(carry, key):
        # Sample actions
        mean, std, *_ = carry
        mets = {}
        eps_key, est_key = jr.split(key)
        eps = jr.normal(
            eps_key, (horizon, num_samples-num_pi_trajs, action_dim)
        )
        sampled_actions = jnp.clip(mean[:, None] + std[:, None] * eps, -1, 1)
        if num_pi_trajs > 0:
            actions = jnp.concat([pi_actions, sampled_actions], axis=1)
        else:
            actions = sampled_actions

        # Compute elite actions
        if plan_method == 'threshold':
            value, global_cost, local_cost = _constrained_estimate_value(
                model, cost_model, est_key, latent, actions, discount,
                cost_discount, use_cm, use_cost
            )
            if use_local_cost:
                C = local_cost
            else:
                C = global_cost
            value, C = jnp.nan_to_num(value), jnp.nan_to_num(C), 
            num_safe_traj = jnp.sum(jax.lax.convert_element_type(C < threshold, jnp.int32))
            
            # extend with extra element to remove after selecting top k: needed to use jit with jnp.nonzero
            value = jnp.concat([jnp.array([-jnp.inf]), value])
            C = jnp.concat([jnp.array([jnp.inf]), C])
            actions = jnp.concat([jnp.zeros_like(actions[:, :1]), actions], axis=1)
            
            def safe_func(values, costs, actions):
                mask = costs < threshold
                safe_elite_idxs = jnp.nonzero(jax.lax.convert_element_type(mask, jnp.int32), size=mask.size)[0]
                return values[safe_elite_idxs], actions[:, safe_elite_idxs]
            elite_value, elite_actions = jax.lax.cond(
                num_safe_traj < num_elites,
                lambda values, costs, actions: (-costs, actions), safe_func,
                value, C, actions
            )
            elite_value, elite_idxs = jax.lax.top_k(elite_value, num_elites)
            elite_actions = elite_actions[:, elite_idxs]
            mask = jnp.ones_like(elite_value)
            mets['num_safe_trajs'] = num_safe_traj
        elif plan_method == 'adaptive':
            value, cost_value, _ = _constrained_estimate_value(
                model, cost_model, est_key, latent, actions, discount,
                cost_discount, use_cm, use_cost
            )
            value, cost_value = jnp.nan_to_num(value), jnp.nan_to_num(cost_value)
            if use_ci_thresholds:
                safe_threshold = cs_h
                value_threshold = qs_l
            else:
                safe_threshold = cost_value[:num_pi_trajs].mean()
                value_threshold = value[:num_pi_trajs].mean()
            
            mask = (cost_value <= safe_threshold) & (value >= value_threshold)
            elite_value = jnp.where(mask, value, value.min())
            elite_actions = actions
            
            if restrict_elites:
                _, m_idxs = jax.lax.top_k(-elite_value, num_samples - num_elites)
                mask = mask.at[m_idxs].set(False)
                elite_value = jnp.where(mask, elite_value, elite_value.min())
            
            num_imprs = jnp.sum(jax.lax.convert_element_type(mask, jnp.int32))
            elite_value, mask = jax.lax.cond(
                num_imprs == 0,
                lambda vs, m: (
                    vs.at[:num_pi_trajs].set(value[:num_pi_trajs]),
                    m.at[:num_pi_trajs].set(True)
                ),
                lambda vs, m: (vs, m),
                elite_value, mask
            )
            mask = mask.astype(jnp.float32)
            mets['no_improvement'] = num_imprs == 0
            mets['improvement'] = num_imprs
            mets['value_threshold'] = value_threshold
            mets['safe_threshold'] = safe_threshold
        elif plan_method == 'unsafe':
            value = jnp.nan_to_num(_estimate_value(
                model, est_key, latent['reward'], actions, discount,
            ))
            elite_value, elite_idxs = jax.lax.top_k(value, num_elites)
            elite_actions = actions[:, elite_idxs]
            mask = jnp.ones_like(elite_value)
        else:
            raise NotImplementedError

        # Update parameters
        max_value = elite_value.max()
        
        score = jnp.exp(temperature * (elite_value - max_value))
        score = score * mask
        score /= (score.sum(0) + 1e-9)
        _mean = jnp.sum(score[None, :, None] * elite_actions, axis=1) / (score.sum(0) + 1e-9)

        std = jnp.clip(
            jnp.sqrt(jnp.sum(score[None, :, None] * (elite_actions - _mean[:, None]) ** 2, axis=1) / (score.sum(0) + 1e-9)), 
            min_std, max_std
        )
        mean = momentum * mean + (1 - momentum) * _mean
        return (mean, std, score, elite_actions), mets

    if plan_method == 'adaptive':
        elite_actions = jnp.empty((horizon, num_samples, action_dim))
        score = jnp.empty(num_samples)
    else:
        elite_actions = jnp.empty((horizon, num_elites, action_dim))
        score = jnp.empty(num_elites)

    i_key, a_key, n_key = jr.split(key, 3)
    init = (mean, std, score, elite_actions)
    last, mets = jax.lax.scan(iteration, init, jr.split(i_key, iterations))
    mean, std, score, elite_actions = last
    
    # Select action
    act_ind = jr.choice(a_key, score.shape[0], p=score)
    actions = elite_actions[:, act_ind]
    a, std = actions[0], std[0]
    if not eval_mode:
        a += std * jr.normal(n_key, (action_dim,))
    a = jnp.clip(a, -1, 1)
    if restrict_action:
        key, sub_key = jr.split(key)
        qs = jnp.squeeze(model.Q(sub_key, current_z, a, 'avg'))
        key, sub_key = jr.split(key)
        cs = jnp.squeeze(model.C(sub_key, current_z, a, 'avg'))
        a = jnp.where((qs >= qs_l) & (cs <= cs_h), a, pi_a)
        metrics['improve_action'] = (qs >= qs_l) & (cs <= cs_h)
    
    mets = jtu.tree_map(lambda x: jnp.mean(x), mets)
    metrics |= mets
    return a, mean, metrics


@eqx.filter_jit
def plan_eval_policy(
    model: WorldModel,
    cost_model: CostModel,
    key: jr.PRNGKey,
    obs: jax.Array,
    prev_mean: jax.Array,
    discount: float,
    cost_discount: float,
    params
):
    print('Compiling plan eval policy ...')
    ekey, akey, ckey = jr.split(key, 3)
    latent = {
        'reward': model.encode(obs, ekey),
        'cost': cost_model.encode(obs, ckey),
    }
    a, prev_mean, _ = plan(
        model, cost_model, akey, latent, prev_mean,
        discount, cost_discount, True,
        **params
    )
    return a, prev_mean


@eqx.filter_jit
def plan_train_policy(
    model: WorldModel,
    cost_model: CostModel,
    key: jr.PRNGKey,
    obs: jax.Array,
    prev_mean: jax.Array,
    discount: float,
    cost_discount: float,
    params
):
    print('Compiling plan train policy ...')
    keys = jr.split(key, 6)
    latent = {
        'reward': model.encode(obs, keys[0]),
        'cost': cost_model.encode(obs, keys[1]),
    }
    a, prev_mean, mets = plan(
        model, cost_model, keys[2], latent, prev_mean,
        discount, cost_discount, False,
        **params
    )
    z = latent['reward']
    if model.use_cost:
        cost = jnp.squeeze(math.two_hot_inv(
            model.cost(keys[3], z, a), model.num_bins, model.vmin, model.vmax
        ), axis=1)
        mets['cost_estd'] = cost.std(0)
    mets['costvalue_estd'] = model.C(keys[4], z, a, return_type='std')
    mets['value_estd'] = model.Q(keys[5], z, a, return_type='std')
    return a, prev_mean, mets


@eqx.filter_jit
def pi_eval_policy(
    model: WorldModel,
    key: jr.PRNGKey,
    obs: jax.Array,
):
    print('Compiling pi eval policy ...')
    ekey, akey = jr.split(key)
    z = model.encode(obs, ekey)
    a = model.pi(akey, z)[0]
    return a


@eqx.filter_jit
def pi_train_policy(
    model: WorldModel,
    key: jr.PRNGKey,
    obs: jax.Array,
):
    print('Compiling pi train policy ...')
    keys = jr.split(key, 5)
    z = model.encode(obs, keys[0])
    a = model.pi(keys[1], z)[1]
    
    mets = {}
    if model.use_cost:
        cost = jnp.squeeze(math.two_hot_inv(
            model.cost(keys[2], z, a), model.num_bins, model.vmin, model.vmax
        ), axis=1)
        mets['cost_estd'] = cost.std(0)
    mets['costvalue_estd'] = model.C(keys[3], z, a, return_type='std')
    mets['value_estd'] = model.Q(keys[4], z, a, return_type='std')
    return a, mets