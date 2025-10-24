import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu

import equinox as eqx
import optax

from spowl.common.buffer import sample_from_buffer
from spowl.spowl.world_model import WorldModel, inference
from spowl.spowl.cost_model import CostModel
from spowl.spowl import math


def _td_target(model: WorldModel, key, next_z, reward, cost, term, discount, cost_discount, cvalue_target):
    pi_key, q_key, c_key = jr.split(key, 3)
    pi = model.pi(pi_key, next_z)[1]
    rtarg = reward + (1 - term) * discount * model.Q(q_key, next_z, pi, return_type='min', target=True)
    ctarg = cost + (1 - term) * cost_discount * model.C(c_key, next_z, pi, return_type=cvalue_target, target=True)
    return rtarg, ctarg


@eqx.filter_jit
def make_step(
    cfg, key, model: WorldModel, data, discount, filter_spec, pi_filter_spec,
    optim, opt_state, scale, pi_optim, pi_opt_state
):
    print('Compiling wm update function ...')
    obs, action, reward, cost, term, done = data
    horizon, batch_size = reward.shape[:2]
    mets = {}
    flat_size = reward.shape[0] * reward.shape[1]
    flatten = lambda x: x.reshape((flat_size, x.shape[-1]))
    
    # Compute targets
    key, enc_key, td_key  = jr.split(key, 3)
    next_z = jax.vmap(model.encode)(flatten(obs[1:]), jr.split(enc_key, flat_size))
    td_targets, ctd_targets = jax.vmap(_td_target, in_axes=(None, 0, 0, 0, 0, 0, None, None, None))(
        model, jr.split(td_key, flat_size), flatten(next_z), flatten(reward), flatten(cost),
        flatten(term), discount, cfg.cost_discount, cfg.cvalue_target
    )
    next_z = next_z.reshape((*reward.shape[:-1], -1))
    td_targets = td_targets.reshape((*reward.shape[:-1], -1))
    ctd_targets = ctd_targets.reshape((*reward.shape[:-1], -1))
    mets['cvalue_targets'] = ctd_targets.mean()
    mets['value_targets'] = td_targets.mean()

    # Prepare for update
    @eqx.filter_value_and_grad(has_aux=True)  
    def full_loss(diff_model, static_model, key):
        md = eqx.combine(diff_model, static_model)
        data_key, init_key, dec_key = jr.split(key, 3) 
        data = (jr.split(data_key, horizon), next_z, action, reward, cost, td_targets, ctd_targets, done[:-1])
        init_z = jax.vmap(md.encode)(obs[0], jr.split(init_key, batch_size))
        
        # Loss for one step in horizon
        def step_loss(z, data):
            key, n_z, a, r, c, td, ctd, d = data
            q_key, cv_key, c_key = jr.split(key, 3) 
            _z = jax.vmap(md.next)(z, a)
            _z = jnp.where(d, n_z, _z)
            qs = jax.vmap(md.Q)(jr.split(q_key, batch_size), z, a)
            cs = jax.vmap(md.C)(jr.split(cv_key, batch_size), z, a)
            reward_preds = jax.vmap(md.reward)(z, a)
            
            _mean = lambda loss: jnp.where(d, 0, loss).sum() / (jnp.size(loss) - d.sum() + 1e-9)
            ens_loss = jax.vmap(
                lambda pred, targ: _mean(math.soft_ce(pred, targ, md.num_bins, md.vmin, md.vmax)), in_axes=(1, None)
            )
            
            losses = {}
            if md.use_cost:
                cost_preds = jax.vmap(md.cost)(jr.split(c_key, batch_size), z, a)
                c_mask = lambda loss, targ: jnp.where(targ > cfg.minimal_cost, loss * cfg.cost_imb_weight, loss)
                cost_loss = jax.vmap(
                    lambda pred, targ: _mean(c_mask(math.soft_ce(pred, targ, md.num_bins, md.vmin, md.vmax), targ)),
                    in_axes=(1, None)
                )
                losses['cost_loss'] = cost_loss(cost_preds, c).mean()
            
            losses['consistency_loss'] = _mean(jnp.square(_z - n_z))
            losses['reward_loss'] = _mean(math.soft_ce(reward_preds, r, md.num_bins, md.vmin, md.vmax))
            losses['value_loss'] = ens_loss(qs, td).mean()
            losses['cvalue_loss'] = ens_loss(cs, ctd).mean()
            return _z, (z, losses)
        
        _z, (zs, losses) = jax.lax.scan(step_loss, init_z, data)
        
        zs = jnp.concat([zs, _z[None]], axis=0)
        rho = jnp.power(cfg.rho, jnp.arange(horizon))
        losses = jax.tree_util.tree_map(lambda x: (x * rho).mean(), losses)
        if md.use_decoder:
            fs = zs.shape[0] * zs.shape[1]
            deco = jax.vmap(md.decode)(zs.reshape((fs, zs.shape[-1])), jr.split(dec_key, fs))
            deco = deco.reshape(obs.shape)
            losses['decoder_loss'] = jnp.square(deco - obs).mean()
        return sum(v * cfg.loss_coefs[k] for k, v in losses.items()), (zs, losses)
    
    # Update model heads
    model = inference(model, False)
    diff_model, static_model = eqx.partition(model, filter_spec)
    key, subkey = jr.split(key)
    (total_loss, (zs, losses)), grads = full_loss(diff_model, static_model, subkey)

    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    mets['total_loss'] = total_loss
    mets.update(losses)
    
    # Update policy
    zs = zs.reshape((-1, zs.shape[-1]))
    
    @eqx.filter_value_and_grad(has_aux=True) 
    def loss(diff_model, static_model, key):
        md = eqx.combine(diff_model, static_model)
        pi_key, q_key, c_key = jr.split(key, 3)
        outs = {}
        _, pis, log_pis, _ = jax.vmap(md.pi)(jr.split(pi_key, zs.shape[0]), zs)
        q_func = lambda k, z, pi: md.Q(k, z, pi, return_type='avg')
        qs = jax.vmap(q_func)(jr.split(q_key, zs.shape[0]), zs, pis)
        qs = qs.reshape((horizon+1, batch_size, -1))
        log_pis = log_pis.reshape((horizon + 1, batch_size, -1))
        new_scale = scale.update(qs[0])
        qs = qs / new_scale
        outs['scale'] = new_scale
    
        # Loss is a weighted sum of Q-values
        rho = jnp.pow(cfg.rho, jnp.arange(len(qs)))
        pi_loss = ((cfg.entropy_coef * log_pis - qs).mean(axis=(1,2)) * rho).mean()
        outs['psi'] = 0
        outs['lambda_'] = model.lagrange_multiplier
        outs['penalty'] = model.penalty_multiplier
        
        if cfg.use_lagrange:
            c_func = lambda k, z, pi: md.C(k, z, pi, return_type=cfg.lag_cost_type)
            cs = jax.vmap(c_func)(jr.split(c_key, zs.shape[0]), zs, pis)
            # Statistics based on batch from buffer (check for multitask)
            g = jnp.mean(cs - cfg.cost_limit)
            lambda_ = model.lagrange_multiplier
            penalty = model.penalty_multiplier
            cond = lambda_ + penalty * g
            
            psi = jnp.where(
                jnp.greater(cond, 0.),
                lambda_ * g + penalty / 2.0 * jnp.power(g, 2),
                -1. / (2. * penalty) * jnp.power(lambda_, 2)
            )
            outs['psi'] = psi
            outs['lambda_'] = jnp.clip(cond, 0, cfg.lagrange_max)
            outs['penalty'] = jnp.clip(penalty * (cfg.penalty_rate + 1.), penalty, 1.)
            pi_loss = pi_loss + psi
            
        return pi_loss, outs
    
    diff_model, static_model = eqx.partition(model, pi_filter_spec)
    key, subkey = jr.split(key)
    (pi_loss, outs), grads = loss(diff_model, static_model, subkey)
    scale = eqx.tree_at(lambda tree: tree._value, scale, replace=outs['scale'])
    updates, pi_opt_state = pi_optim.update(grads, pi_opt_state)
    model = eqx.apply_updates(model, updates)
    model = eqx.tree_at(
        lambda tree: (tree.lagrange_multiplier, tree.penalty_multiplier), model,
        (outs['lambda_'], outs['penalty'])
    )
    mets['pi_loss'] = pi_loss
    mets['pi_scale'] = scale.value
    mets['psi'] = outs['psi']
    
    # Update target Q-functions
    soft_upd = lambda p_targ, p: p_targ + cfg.tau * (p - p_targ) if eqx.is_inexact_array(p) else p_targ
    model = eqx.tree_at(
        lambda tree: tree._target_Qs, model, replace_fn=lambda target_tree: jtu.tree_map(
            soft_upd, target_tree, model._Qs
    ))
    model = eqx.tree_at(
        lambda tree: tree._target_Cs, model, replace_fn=lambda target_tree: jtu.tree_map(
            soft_upd, target_tree, model._Cs
    ))
    
    # Return training statistics
    model = inference(model, True)
    return mets, model, opt_state, scale, pi_opt_state



@eqx.filter_jit
def _upd(
    cfg, key, is_full, step_id, buffer_state,
    model, opt_state, scale, pi_opt_state,
    batch_size, horizon, discount, filter_spec,
    pi_filter_spec, optim, pi_optim,
):  
    print('Compiling agent single update function ...')
    s_key, u_key = jr.split(key)
    sample = sample_from_buffer(
        s_key, batch_size, horizon, buffer_state, is_full, step_id
    )
    mets, model, opt_state, scale, pi_opt_state = make_step(
        cfg, u_key, model, sample, discount, filter_spec, pi_filter_spec,
        optim, opt_state, scale, pi_optim, pi_opt_state
    )
    
    mets.update({
        "lagrange_multiplier": model.lagrange_multiplier,
        "penalty_multiplier": model.penalty_multiplier,
    })
    return mets, model, opt_state, scale, pi_opt_state


@eqx.filter_jit
def multi_upd(
    cfg, key, is_full, step_id, buffer_state, model, opt_state, scale, pi_opt_state,
    batch_size, horizon, discount, filter_spec, pi_filter_spec, optim, pi_optim,
    num_updates
):
    print('Compiling agent update function ...')
    dynamic_wm, static_wm = eqx.partition(model, eqx.is_array)
    dynamic_scale, static_scale = eqx.partition(scale, eqx.is_array)

    def _step(carry, key):
        dynamic_wm, opt_state, dynamic_scale, pi_opt_state = carry
        model = eqx.combine(dynamic_wm, static_wm)
        scale = eqx.combine(dynamic_scale, static_scale)
        mets, model, opt_state, scale, pi_opt_state = _upd(
            cfg, key, is_full, step_id, buffer_state, model, opt_state, scale, pi_opt_state,
            batch_size, horizon, discount, filter_spec, pi_filter_spec, optim, pi_optim
        )
        dynamic_wm = eqx.filter(model, eqx.is_array)
        dynamic_scale = eqx.filter(scale, eqx.is_array)
        return (dynamic_wm, opt_state, dynamic_scale, pi_opt_state), mets
    
    _init = (dynamic_wm, opt_state, dynamic_scale, pi_opt_state)
    out, mets = jax.lax.scan(_step, _init, jr.split(key, num_updates))
    (dynamic_wm, opt_state, dynamic_scale, pi_opt_state) = out
    model = eqx.combine(dynamic_wm, static_wm)
    scale = eqx.combine(dynamic_scale, static_scale)
    return mets, model, opt_state, scale, pi_opt_state


@eqx.filter_jit
def make_cm_step(
    key: jr.PRNGKey,
    data: tuple[jax.Array],
    cost_model: CostModel,
    c_optim, c_opt_state, cfg
):
    # Train on 1step transitions
    print('Compiling cost model single update function ...')
    obs, action, reward, cost, term, done = data
    done = done[:-1]
    
    # Compute cost model
    @eqx.filter_value_and_grad(has_aux=True)
    def cm_loss(diff_model, static_model):
        metrics = {}
        cost_md: CostModel = eqx.combine(diff_model, static_model)
        state_error, pred_costs = jax.vmap(cost_md.rollout, in_axes=(1, 1, 1, 0), out_axes=1)(
            obs, action, done, jr.split(key, obs.shape[1])
        )
        
        # consider episode ends
        mask = jnp.logical_not(done)
        denom = jnp.sum(mask)
        
        unsafe_mask = (cost > cfg.cost_threshold) & mask
        safe_mask = (cost <= cfg.cost_threshold) & mask
        
        # weights for unsafe states
        if cfg.imbalance_weight > 0:
            mask = jnp.where(unsafe_mask, cfg.imbalance_weight * mask, mask)
            
        loss = jnp.square(pred_costs - cost)
        loss = mask * loss

        loss = jnp.sum(loss) / denom
        metrics['cost_loss'] = loss
        equals = jnp.isclose(cost, pred_costs, rtol=0, atol=cfg.tol)
        metrics['accuracy'] = jnp.sum(equals * jnp.logical_not(done)) / denom
        metrics['unsafe_accuracy'] = jnp.sum(equals * unsafe_mask) / (unsafe_mask.sum() + 1e-9)
        metrics['unsafe_transitions'] = unsafe_mask.sum()
        metrics['safe_accuracy'] = jnp.sum(equals * safe_mask) / (safe_mask.sum() + 1e-9)
        metrics['safe_transitions'] = safe_mask.sum()
        
        metrics['minval'] = pred_costs.min()
        metrics['maxval'] = pred_costs.max()
        loss = loss * cfg.cost_weight
        
        predict_loss = jnp.sum(jnp.where(done, 0, state_error)) / denom
        metrics['dyn_loss'] = predict_loss
        loss = loss + cfg.predict_weight * predict_loss
        
        metrics['total_loss'] = loss
        return loss, metrics
        
    cost_model = eqx.nn.inference_mode(cost_model, False)
    diff_model, static_model = eqx.partition(cost_model, eqx.is_inexact_array)
    (cost_loss, mets), grads = cm_loss(diff_model, static_model)

    updates, c_opt_state = c_optim.update(grads, c_opt_state)
    cost_model = eqx.apply_updates(cost_model, updates)
    cost_model = eqx.nn.inference_mode(cost_model, True)
    mets['grad_norm'] = optax.global_norm(updates)
    mets['mean_cost'] = cost.mean()
    return mets, cost_model, c_opt_state


@eqx.filter_jit
def update_cm(cfg, key, is_full, step_id, buffer_state, cost_model, optim, opt_state):
    print("Compiling cost model update function ...")
    dynamic_cm, static_cm = eqx.partition(cost_model, eqx.is_array)
    
    def _cm_upd(carry, key):
        dynamic_cm, opt_state = carry
        cost_model = eqx.combine(dynamic_cm, static_cm)
        sample_key, step_key = jr.split(key)
        sample = sample_from_buffer(
            sample_key, cfg.batch_size, cfg.horizon, buffer_state, is_full, step_id
        )
        mets, cost_model, opt_state = make_cm_step(
            step_key, sample, cost_model, optim, opt_state, cfg
        )
        dynamic_cm = eqx.filter(cost_model, eqx.is_array)
        return (dynamic_cm, opt_state), mets
        
    _init = (dynamic_cm, opt_state)
    (dynamic_cm, opt_state), mets = jax.lax.scan(_cm_upd, _init, jr.split(key, cfg.n_updates))
    cost_model = eqx.combine(dynamic_cm, static_cm)
    mets = jtu.tree_map(lambda x: x.mean(), mets)
    return cost_model, opt_state, mets