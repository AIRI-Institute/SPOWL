from omegaconf import DictConfig

from spowl.common.types import EnvironmentFactory


def get_domain_and_task(cfg: DictConfig) -> tuple[str, DictConfig]:
    assert len(cfg.environment.keys()) == 1
    domain_name, task = list(cfg.environment.items())[0]
    return domain_name, task


def make(cfg: DictConfig) -> EnvironmentFactory:
    assert len(cfg.environment.keys()) == 1
    domain_name, task_config = get_domain_and_task(cfg)
    if domain_name == "safety_gymnasium":
        from spowl.benchmarks.safety_gymnasium import make
        make_env = make(cfg)
    else:
        raise NotImplementedError(f"Environment {domain_name} not implemented")
    return make_env