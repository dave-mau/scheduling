from computation_sim.basic_types import Time
from computation_sim.time import Clock
import computation_sim.system as system
from .reward import Reward
from .types import SystemCollection
import gymnasium as gym
import numpy as np
from typing import List

class HierarchicalSystemBase(gym.Env):
    def __init__(
        self,
        clock: Clock,
        system_collection: SystemCollection,
        reward: Reward,
        dt: Time = 10,
    ):
        # Store init params
        self.clock: Clock = clock
        self._system_collection: SystemCollection = system_collection
        self._reward: Reward = reward
        self.dt = dt

        # Set dimensionality of action / observation spaces
        system_collection.system.update()
        self.action_space = gym.spaces.Discrete(system.num_actions(self.system.num_action))
        lb = -np.inf * np.ones((len(self.system.state),), dtype=float)
        ub = +np.inf * np.ones((len(self.system.state),), dtype=float)
        self.observation_space = gym.spaces.Box(lb, ub, dtype=float)

    @property
    def system(self) -> system.System:
        return self._system_collection.system

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

    def act(self, action: List[int]):
        self.system.act(action)

    def advance(self):
        self.clock += self.dt
        self.system.update()

    def step(self, action: int):
        # Reset the sinks that count number of lost messages
        # This means, we count the number of lost messages from now on.
        for sink in self._system_collection.sinks:
            sink.reset()

        # Set baseline for reward function that
        self._reward.record_basline_state()

        # Apply the action and advance the system
        action = system.unpack_action(self.system.num_action, action)
        self.act(action)
        self.advance()

        # Compute the reward
        reward, reward_info = self._reward.compute(action)

        # Build the reward
        return self.state, reward, False, False, reward_info

    def _count_lost_msgs(self) -> int:
        return
