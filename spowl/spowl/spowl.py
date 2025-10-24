from collections import defaultdict

import optax
import numpy as np
import equinox as eqx
import jax.numpy as jnp
from gymnasium.spaces import Box
from omegaconf import DictConfig

from spowl.common.types import FloatArray
from spowl.common.buffer import Buffer
from spowl.common.trajectory import TrajectoryData
from spowl.common.utils import PRNGSequence
from spowl.spowl.world_model import make_world_model
from spowl.spowl.cost_model import make_cost_model
from spowl.spowl.scale import RunningScale
from spowl.spowl import policies
from spowl.spowl.update import multi_upd, update_cm


class SPOWL:
    def __init__(
        self,
        observation_space: Box,
        action_space: Box,
        config: DictConfig,
    ):
        self.config = config
        self.action_space = action_space
        self.action_space.seed(config.training.seed)
        
        self.prng = PRNGSequence(config.training.seed)
        
        self.replay_buffer = Buffer(
            observation_space.shape,
            action_space.shape,
            config.agent.replay_buffer.capacity // config.training.parallel_envs,
            config.training.parallel_envs
        )

        # Initialize World Model
        self.model, wm_opt, pi_opt = make_world_model(
            config.training.seed, observation_space.shape[0], action_space.shape[0],
            config.agent.world_model, config.agent.opt
        )
        self.filter_spec, self.optim, self.opt_state = wm_opt
        self.pi_filter_spec, self.pi_optim, self.pi_opt_state = pi_opt
        
        # Initialize Cost Model
        cm_config = config.agent.cost_model
        self.cost_model = make_cost_model(
            config.training.seed,
            cm_config.use_custom_init,
            cm_config.last_zero,
            cm_config.cost_mlp_act,
            cm_config.cm_dropout,
            cm_config.cm_enc_dim,
            observation_space.shape[0],
            action_space.shape[0],
            cm_config.cm_enc_layers,
            cm_config.cm_enc_act,
            cm_config.cm_simnorm_dim,
            cm_config.cm_state_dim,
            cm_config.cm_dim,
            cm_config.cm_layers,
            cm_config.cm_dyn_act,
        )
        self.c_optim = optax.chain(
            optax.clip_by_global_norm(config.agent.opt.grad_clip_norm),
            optax.adam(learning_rate=config.agent.opt.cost_lr),
        )
        c_state = eqx.filter(self.cost_model, eqx.is_inexact_array)
        self.c_opt_state = self.c_optim.init(c_state)
        self.cost_model = eqx.nn.inference_mode(self.cost_model, True)
        
        self.scale = RunningScale(config.training.tau)
        self.discount = config.training.gamma
        self.cost_discount = config.upd.cost_discount
        self._prev_mean = jnp.zeros((config.training.parallel_envs, config.agent.planning.horizon, action_space.shape[0]))
        self._eval_prev_mean = jnp.zeros((config.training.parallel_envs, config.agent.planning.horizon, action_space.shape[0]))
        self.mets = defaultdict(list)
        self.train_step = 0

    def __call__(
        self,
        observation: FloatArray,
        train: bool,
        step: int,
        is_first: FloatArray,
    ) -> FloatArray:
        
        # Initial train exploration
        if train and step < self.config.training.exploration_steps or self.replay_buffer.volume == 0:
            actions = np.asarray([self.action_space.sample() for _ in range(self.config.training.parallel_envs)])
        else:
            actions = self.act(observation, is_first, not train)
            
        if train and self.replay_buffer.volume > 0 and step > self.train_step + self.config.training.train_every:
            self.train_step = step
            self.update()
        return actions
    
    def add_data(self, transitions: TrajectoryData, start: int, end: int):
        self.replay_buffer.add(transitions, start, end)
        
    def act(self, obs, is_first, eval_mode):
        obs = jnp.array(obs)

        if eval_mode and self.config.agent.policy.eval == "plan":
            t0 = jnp.array(is_first)[:, None, None]
            self._eval_prev_mean = t0 * jnp.zeros_like(self._eval_prev_mean) + (1 - t0) * self._eval_prev_mean
            jit_eval_policy = eqx.filter_vmap(
                policies.plan_eval_policy, in_axes=(None, None, 0, 0, 0, None, None, None)
            )
            a, self._eval_prev_mean = jit_eval_policy(
                self.model, self.cost_model, self.prng.take_n(obs.shape[0]),
                obs, self._eval_prev_mean, self.discount, self.cost_discount,
                self.config.agent.planning
            )
        if eval_mode and self.config.agent.policy.eval == "pi":
            jit_eval_policy = eqx.filter_vmap(policies.pi_eval_policy, in_axes=(None, 0, 0))
            a = jit_eval_policy(self.model, self.prng.take_n(obs.shape[0]), obs)
            
        if (not eval_mode) and self.config.agent.policy.train == "plan":
            t0 = jnp.array(is_first)[:, None, None]
            self._prev_mean = t0 * jnp.zeros_like(self._prev_mean) + (1 - t0) * self._prev_mean
            jit_train_policy = eqx.filter_vmap(
                policies.plan_train_policy, in_axes=(None, None, 0, 0, 0, None, None, None)
            )
            a, self._prev_mean, mets = jit_train_policy(
                self.model, self.cost_model, self.prng.take_n(obs.shape[0]),
                obs, self._prev_mean, self.discount, self.cost_discount,
                self.config.agent.planning
            )
            for k, v in mets.items():
                self.mets[k].append(float(jnp.mean(v)))
        if (not eval_mode) and self.config.agent.policy.train == "pi":
            jit_train_policy = eqx.filter_vmap(policies.pi_train_policy, in_axes=(None, 0, 0))
            a, mets = jit_train_policy(self.model, self.prng.take_n(obs.shape[0]), obs)
            for k, v in mets.items():
                self.mets[k].append(float(jnp.mean(v)))

        assert not jnp.any(jnp.isnan(a))
        a = np.array(a)
        return a

    def update(self):
        mets, self.model, self.opt_state, self.scale, self.pi_opt_state = multi_upd(
            self.config.upd, next(self.prng), self.replay_buffer.is_full, jnp.int32(self.replay_buffer.step_id), self.replay_buffer._state,
            self.model, self.opt_state, self.scale, self.pi_opt_state,
            self.config.upd.batch_size, self.config.agent.planning.horizon, self.discount, self.filter_spec,
            self.pi_filter_spec, self.optim, self.pi_optim, self.config.upd.num_steps_per_update,
        )
        
        for k, v in mets.items():
            self.mets[k].append(float(jnp.mean(v)))
        
        if self.config.agent.cost_model.upd:
            self.cost_model, self.c_opt_state, c_mets = update_cm(
                self.config.agent.cost_model_upd, next(self.prng), self.replay_buffer.is_full, jnp.int32(self.replay_buffer.step_id), self.replay_buffer._state,
                self.cost_model, self.c_optim, self.c_opt_state
            )
            for k, v in c_mets.items():
                self.mets[f'cost_model/{k}'].append(float(jnp.mean(v)))