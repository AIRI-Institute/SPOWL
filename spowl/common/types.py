from dataclasses import dataclass, field
from typing import (
    Callable,
    Protocol,
    Union,
)

import jax
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from numpy import typing as npt
from omegaconf import DictConfig


FloatArray = npt.NDArray[Union[np.float32, np.float64]]

EnvironmentFactory = Callable[[], Union[Env[Box, Box], Env[Box, Discrete]]]

Policy = Union[Callable[[jax.Array, jax.Array | None], jax.Array], jax.Array]


@dataclass
class Report:
    metrics: dict[str, float]
    videos: dict[str, npt.ArrayLike] = field(default_factory=dict)


class Agent(Protocol):
    config: DictConfig
