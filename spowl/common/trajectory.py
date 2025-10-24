from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np
from numpy import typing as npt


class Transition(NamedTuple):
    observation: npt.NDArray[Any]
    action: npt.NDArray[Any]
    reward: npt.NDArray[Any]
    cost: npt.NDArray[Any]
    terminated: npt.NDArray[Any]
    truncated: npt.NDArray[Any]


TrajectoryData = Transition


@dataclass
class Trajectory:
    transitions: TrajectoryData
    index: int = 1
    frames: list[npt.NDArray[np.float32 | np.int8]] = field(default_factory=list)
    
    def add_transition(self, observation, action, reward, cost, terminated, truncated):
        self.transitions.observation[:, self.index] = observation
        self.transitions.action[:, self.index] = action
        self.transitions.reward[:, self.index] = reward[:, None]
        self.transitions.cost[:, self.index] = cost[:, None]
        self.transitions.terminated[:, self.index] = terminated[:, None]
        self.transitions.truncated[:, self.index] = truncated[:, None]
        self.index += 1