from omegaconf import DictConfig

from spowl.benchmarks import get_domain_and_task
from spowl.common.types import EnvironmentFactory


def make(cfg: DictConfig) -> EnvironmentFactory:
    def make_env():
        import safety_gymnasium
        from safety_gymnasium.wrappers import SafetyGymnasium2Gymnasium

        _, task_cfg = get_domain_and_task(cfg)
        if task_cfg.task is not None:
            task = task_cfg.task
        else:
            raise ValueError("Environment task is not defined")
        env = safety_gymnasium.make(task, render_mode='rgb_array', camera_name='fixedfar', width=128, height=128)
        env = SafetyGymnasium2Gymnasium(env)
        return env

    return make_env