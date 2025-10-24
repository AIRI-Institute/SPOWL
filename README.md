# Safe Planning and Policy Optimization via World Model Learning

[![arXiv](https://img.shields.io/badge/arXiv-2506.04828-b31b1b.svg)](https://arxiv.org/abs/2506.04828)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Equinox](https://img.shields.io/badge/Framework-Equinox-ffca28.svg)](https://docs.kidger.site/equinox/)
[![Conda](https://img.shields.io/badge/Environment-miniconda-blue.svg)](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions)
[![DOI](https://img.shields.io/badge/doi-10.3233%2FFAIA251029-green?style=flat)](https://dx.doi.org/10.3233/FAIA251029)

---

## Overview

**SPOWL** (*Safe Planning and Policy Optimization via World Model Learning*) is a framework for **safe reinforcement learning** that unifies **world model learning** and **policy optimization**. It leverages latent-space dynamics modeling and constrained optimization to achieve safe and efficient learning in complex continuous control environments.

---

## Requirements

- **Python**
- **[miniconda/conda](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions)**

---

## Installation

Get started with SPOWL:

1. Create a conda environment:
   ```bash
   conda create -n spowl python==3.10
   ```
2. Activate the environment:
   ```bash
   conda activate spowl
   ```
3. Install Safety Gymnasium
    ```bash
    wget https://github.com/PKU-Alignment/safety-gymnasium/archive/refs/heads/main.zip
    unzip main.zip
    rm -rf main.zip
    pip install -e safety-gymnasium-main
    ```
3. Install jax:
   ```bash
   pip install --no-cache-dir --upgrade pip
   pip install --no-cache-dir --upgrade "jax[cuda12]"
   ```
4. Install other requirements:
   ```bash
   pip install --no-cache-dir hydra-core tabulate wandb tqdm moviepy equinox optax
   ```
5. Install for 'osmesa':
   ```bash
   conda install -c conda-forge mesalib
   ```

6. Fix dependencies:
   ```bash
   pip install --no-cache-dir gymnasium-robotics==1.2.3 numpy==1.25.0
   ```

---

## Usage

Run the training script to display all available options and configurations with:
```bash
python train.py --help
```

Run the training script to train default SPOWL configuration:
```bash
python train.py
```

---

## SPOWL in some tasks

### Point Goal 1
![Point Goal 1](media/point_goal1.gif)

### Point Goal 2
![Point Goal 2](media/point_goal2.gif)

### Point Button 1
![Point Button 1](media/point_button1.gif)

### Point Push 1
![Point Push 1](media/point_push1.gif)

### Car Goal 1
![Car Goal 1](media/car_goal1.gif)

### Doggo Goal 1
![Doggo Goal 1](media/doggo_goal1.gif)

### Ant Goal 1
![Ant Goal 1](media/ant_goal1.gif)

---

## Citation

If you use SPOWL in your research, please cite:

```bibtex
@article{latyshev2025spowl,
  title={Safe Planning and Policy Optimization via World Model Learning},
  author={Latyshev, Artem and Gorbov, Gregory and Panov, Aleksandr I.},
  journal={arXiv preprint arXiv:2506.04828},
  year={2025}
}