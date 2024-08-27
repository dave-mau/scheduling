import gymnasium as gym
import numpy as np
from computation_sim.basic_types import Time
from computation_sim.nodes import ConstantNormalizer, SinkNode, OutputNode
from computation_sim.system import num_actions, unpack_action
from computation_sim.time import (
    Clock,
    GammaDistributionSampler,
    GaussianTimeSampler,
    as_age,
)
from system import HierarchicalSystemBuilder
from typing import List


class MultiStageEnv(gym.Env):
    def __init__(
        self,
        clock: Clock,
        system: HierarchicalSystemBuilder,
        sinks: List[SinkNode],
        output: OutputNode,
        dt: Time = 10,
        cost_message_loss=1.0,
        cost_output_time=0.1,
        cost_input=0.01,
    ):
        # Store init params
        self.clock = clock
        self.system = system
        self.sinks = sinks
        self.output = output
        
        self.dt = dt
        self.cost_message_loss = cost_message_loss
        self.cost_output_time = cost_output_time
        self.cost_input = cost_input

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
        info = dict(
            lost_messages=float(self._count_lost_msgs()),
            output_age_min=0,
            output_age_max=0,
            output_age_avg=0,
        )
        reward = -self.cost_message_loss * info["lost_messages"]
        if self._output.last_received:
            info["output_age_max"] = as_age(
                self._output.last_received.header.t_measure_oldest,
                self.clock.get_time(),
            )
            info["output_age_min"] = as_age(
                self._output.last_received.header.t_measure_youngest,
                self.clock.get_time(),
            )
            info["output_age_avg"] = as_age(
                self._output.last_received.header.t_measure_average,
                self.clock.get_time(),
            )
            reward -= self.cost_output_time * float(info["output_age_max"])
        reward -= self.cost_input * np.sum(action)

        return self.state, reward, False, False, info

    def _count_lost_msgs(self) -> int:
        return sum(len(sink.received_messages) for sink in self.sinks)
