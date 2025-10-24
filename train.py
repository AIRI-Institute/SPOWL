import os
os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import hydra
from omegaconf import OmegaConf

from spowl.common.trainer import get_state_path, load_state, start_fresh


@hydra.main(config_name='config', config_path='spowl/configs')
def main(cfg):
    print(
        f"Setting up experiment with the following configuration: "
        f"\n{OmegaConf.to_yaml(cfg)}"
    )
    state_path = get_state_path()
    if os.path.exists(state_path):
        print(f"Resuming experiment from: {state_path}")
        trainer = load_state(cfg, state_path)
    else:
        print("Starting a new experiment.")
        trainer = start_fresh(cfg)
    with trainer:
        trainer.train()
    print("Done training.")


if __name__ == '__main__':
    main()
