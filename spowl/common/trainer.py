import os
import time
from typing import Optional

import cloudpickle
from omegaconf import DictConfig
import numpy as np

from spowl import benchmarks
from spowl.spowl.spowl import SPOWL
from spowl.common.types import EnvironmentFactory, Report
from spowl.common.utils import PRNGSequence
from spowl.common.logging import TrainingLogger, StateWriter
from spowl.common.episodic_async_env import EpisodicAsync 
from spowl.common import driver


_TRAINING_STATE = "state.pkl"


def get_trainer(name):
    trainers = {"online": OnlineTrainer}
    if name in trainers:
        return trainers[name]
    else:
        raise NotImplementedError(f"Unknown trainer type: {name}")


def get_state_path() -> str:
    log_path = os.getcwd()
    state_path = os.path.join(log_path, _TRAINING_STATE)
    return state_path


def load_state(cfg, state_path) -> "OnlineTrainer":
    return get_trainer(cfg.training.trainer).from_pickle(cfg, state_path)


def start_fresh(
    cfg: DictConfig,
) -> "OnlineTrainer":
    make_env = benchmarks.make(cfg)
    return get_trainer(cfg.training.trainer)(cfg, make_env)


class OnlineTrainer:
    def __init__(
        self,
        config: DictConfig,
        make_env: EnvironmentFactory,
        agent: SPOWL | None = None,
        start_epoch: int = 0,
        step: int = 0,
        seeds: PRNGSequence | None = None,
        total_cost: float = 0,
    ):
        self.config = config
        self.make_env = make_env
        self.epoch = start_epoch
        self.step = step
        self.seeds = seeds
        self.logger: TrainingLogger | None = None
        self.state_writer: StateWriter | None = None
        self.env: EpisodicAsync | None = None
        self.agent = agent
        self.total_cost = total_cost
    
    def __enter__(self):
        log_path = os.getcwd()
        self.logger = TrainingLogger(self.config)
        self.state_writer = StateWriter(log_path, _TRAINING_STATE)
        self.env = EpisodicAsync(
            self.make_env,
            self.config.training.parallel_envs,
            self.config.training.time_limit,
            self.config.training.action_repeat,
        )
        if self.seeds is None:
            self.seeds = PRNGSequence(self.config.training.seed)
        if self.agent is None:
            self.agent = self.make_agent()
        return self
    
    def make_agent(self) -> SPOWL:
        assert self.env is not None
        if self.config.agent.name == "spowl":
            agent = SPOWL(
                self.env.observation_space,
                self.env.action_space,
                self.config,
            )
        else:
            raise NotImplementedError(f"Unknown agent type: {self.config.agent.name}")
        return agent
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self.logger is not None and self.state_writer is not None
        self.state_writer.close()
    
    def train(self, epochs: Optional[int] = None) -> None:
        start = self.epoch
        for epoch in range(start, epochs or self.config.training.epochs):
            print(f"Training epoch #{epoch}")
            summary, wall_time, steps = self._run_training_epoch(
                self.config.training.episodes_per_epoch
            )
            self.total_cost += summary.metrics['total_cost']
            metrics = {
                "train/reward_return": summary.metrics['reward_return'],
                "train/cost_return": summary.metrics['cost_return'],
                "train/cost_rate": self.total_cost / self.step,
                "train/fps": steps / wall_time,
            }
            
            metrics |= {
                f'agent/{k}': np.mean(v) for k, v in self.agent.mets.items()
            }
            self.agent.mets.clear()
            
            summary, wall_time, steps = self._run_eval()
            metrics |= {
                **summary.metrics,
                "eval/fps": steps / wall_time,
            }
            
            metrics |= {
                f'buffer/{k}': v for k, v in self.agent.replay_buffer.get_metrics().items()
            }
            self.logger.log(metrics, self.step)
            for k, v in summary.videos.items():
                self.logger.log_video(v, self.step, k)
            self.epoch = epoch + 1
            self.state_writer.write(self.state)
            
    def _run_training_epoch(self, episodes_per_epoch: int) -> tuple[Report, float, int]:
        start_time = time.time()
        self.env.reset(seed=int(next(self.seeds)[0].item()))
        summary, step = driver.epoch(
            self.agent, self.env, episodes_per_epoch, True,
            self.step, self.config.training.render_episodes,
        )
        steps = step - self.step
        self.step = step
        next(self.seeds)
        end_time = time.time()
        wall_time = end_time - start_time
        return summary, wall_time, steps
    
    def _run_eval(self) -> tuple[Report, float, int]:
        start_time = time.time()
        self.env.reset(seed=int(next(self.seeds)[0].item()))
        summary, steps = driver.epoch(
            self.agent, self.env, 1, False,
            0, self.config.training.render_episodes,
        )
        next(self.seeds)
        end_time = time.time()
        wall_time = end_time - start_time
        
        metrics = {}
        for k, v in summary.metrics.items():
            metrics |= {
                f"eval/{k}_{i}": val for i, val in enumerate(v) 
            }
            metrics |= {
                f"eval/mean_{k}": v.mean(),
                f"eval/std_{k}": v.std(),
            }
            
        summary.metrics = metrics
        return summary, wall_time, steps
    
    @classmethod
    def from_pickle(cls, config: DictConfig, state_path: str) -> "OnlineTrainer":
        with open(state_path, "rb") as f:
            make_env, seeds, agent, epoch, step, total_cost = cloudpickle.load(f).values()
        assert agent.config == config, "Loaded different hyperparameters."
        print(f"Resuming from step {step}")
        return cls(
            config=agent.config,
            make_env=make_env,
            start_epoch=epoch,
            seeds=seeds,
            agent=agent,
            step=step,
            total_cost=total_cost,
        )
    
    @property
    def state(self):
        return {
            "make_env": self.make_env,
            "seeds": self.seeds,
            "agent": self.agent,
            "epoch": self.epoch,
            "step": self.step,
            "total_cost": self.total_cost
        }