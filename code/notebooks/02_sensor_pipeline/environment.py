import gymnasium as gym
import numpy as np
from computation_sim.basic_types import Time
from computation_sim.nodes import ConstantNormalizer, SinkNode, OutputNode
from computation_sim.system import num_actions, unpack_action, System
from computation_sim.time import Clock
from typing import List, Callable, Dict, Tuple


class MultiStageEnv(gym.Env):
    def __init__(
        self,
        clock: Clock,
        system: System,
        sinks: List[SinkNode],
        output: OutputNode,
        compute_reward:Callable[["MultiStageEnv", np.ndarray], Tuple[float, Dict]],
        dt: Time = 10,
        
    ):
        # Store init params
        self.clock: Clock = clock
        self.system: System = system
        self.sinks: List[SinkNode] = sinks
        self.output: OutputNode = output
        
        self.dt = dt
        self.compute_reward = compute_reward

        # Build system
        self.age_normalizer = ConstantNormalizer(100.0)
        self.count_normalizer = ConstantNormalizer(1.0)
        self.occupancy_normalizer = ConstantNormalizer(1.0)

        # Set dimensionality of action / observation spaces
        self.action_space = gym.spaces.Discrete(num_actions(self.system.num_action))
        lb = -np.inf * np.ones((len(self.system.state),), dtype=float)
        ub = +np.inf * np.ones((len(self.system.state),), dtype=float)
        self.observation_space = gym.spaces.Box(lb, ub, dtype=float)

    @property
    def time(self) -> Time:
        return self.clock.get_time()

    @property
    def state(self) -> np.ndarray:
        return np.array(self.system.state).flatten()

    def reset(self, seed=None):
        super().reset(seed=seed)
        np.random.seed(seed)
        self.clock.reset()
        self.system.reset()
        self.system.update()
        return self.state, {}

    def step(self, action: int):
        # Reset the sinks that count number of lost messages
        # This means, we count the number of lost messages from now on.
        for sink in self.sinks:
            sink.reset()

        # Get the action vector from the action id
        action = unpack_action(self.system.num_action, action)
        self.system.act(action)
        self.clock += self.dt
        self.system.update()

        # Build the reward
        reward, info = self.compute_reward(action)
        return self.state, reward, False, False, info

    def _count_lost_msgs(self) -> int:
        return 
