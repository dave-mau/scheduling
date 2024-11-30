from typing import Callable, Dict, Iterable, List, NamedTuple, Tuple

import gymnasium as gym
import numpy as np
from computation_sim.basic_types import Header, Time
from computation_sim.nodes import (
    ConstantNormalizer,
    FilteringMISONode,
    OutputNode,
    SinkNode,
)
from computation_sim.system import System, num_actions, unpack_action
from computation_sim.time import Clock, TimeProvider, as_age
from three_stage_system_builder import (
    FilterinMISOActionCollection,
    MultiStageSystemCollection,
)


class Reward:
    def __init__(
        self,
        time_provider: TimeProvider,
        system_collection: MultiStageSystemCollection,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
    ):
        self.time_provider = time_provider
        self.t_init = self.time_provider.time
        self.system_collection = system_collection
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._inactive_action_collections: List[FilterinMISOActionCollection] = list()

    def record_basline_state(self):
        self._inactive_action_collections = list(
            filter(lambda x: not x.node.is_busy, self.system_collection.action_collections)
        )

    def _count_vanished_msgs(self) -> Dict[str, int]:
        return {sink.id: sink.count for sink in self.system_collection.sinks}

    def _count_rejected_msgs(self) -> Dict[str, int]:
        counts = dict()
        activated_collections = filter(lambda x: x.node.is_busy, self._inactive_action_collections)
        for collection in activated_collections:
            counts[collection.node.id] = len(collection.input_buffers) - collection.node.filtered_input_count
        return counts

    def _get_output_ages(self) -> Header:
        last_received = self.system_collection.output.last_received
        now = self.time_provider.time
        if last_received:
            return dict(
                output_age_min=(as_age(last_received.header.t_measure_youngest, now)),
                output_age_max=(as_age(last_received.header.t_measure_oldest, now)),
                output_age_avg=(as_age(last_received.header.t_measure_average, now)),
            )
        else:
            return dict(
                output_age_min=(as_age(self.t_init, now)),
                output_age_max=(as_age(self.t_init, now)),
                output_age_avg=(as_age(self.t_init, now)),
            )

    def compute(self, action: List[int]) -> Tuple[float, dict]:
        info = dict(
            lost_messages=sum(self._count_vanished_msgs().values()),
            rejected_messages=sum(self._count_rejected_msgs().values()),
            num_activations=np.sum(action),
        )
        info.update(self._get_output_ages())
        reward = -self._alpha * float(info["lost_messages"] + info["rejected_messages"])
        reward -= self._beta * float(info["output_age_max"])
        reward -= self._gamma * float(info["num_activations"])
        return reward, info


class MultiStageEnv(gym.Env):
    def __init__(
        self,
        clock: Clock,
        system_collection: MultiStageSystemCollection,
        reward: Reward,
        dt: Time = 10,
    ):
        # Store init params
        self.clock: Clock = clock
        self._system_collection: MultiStageSystemCollection = system_collection
        self._reward: Reward = reward
        self.dt = dt

        # Set dimensionality of action / observation spaces
        system_collection.system.update()
        self.action_space = gym.spaces.Discrete(num_actions(self.system.num_action))
        lb = -np.inf * np.ones((len(self.system.state),), dtype=float)
        ub = +np.inf * np.ones((len(self.system.state),), dtype=float)
        self.observation_space = gym.spaces.Box(lb, ub, dtype=float)

    @property
    def system(self) -> System:
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
        action = unpack_action(self.system.num_action, action)
        self.act(action)
        self.advance()

        # Compute the reward
        reward, reward_info = self._reward.compute(action)

        # Build the reward
        return self.state, reward, False, False, reward_info

    def _count_lost_msgs(self) -> int:
        return
