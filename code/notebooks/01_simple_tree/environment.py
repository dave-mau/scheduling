import gymnasium as gym
import numpy as np
from computation_sim.basic_types import Time
from computation_sim.example_systems import SimpleTreeBuilder
from computation_sim.nodes import ConstantNormalizer
from computation_sim.time import (
    Clock,
    GammaDistributionSampler,
    GaussianTimeSampler,
    as_age,
)


class TreeEnv(gym.Env):
    def __init__(
        self,
        num_sensors: int = 5,
        dt: Time = 10,
        cost_message_loss=1.0,
        cost_output_time=0.1,
        cost_input=0.01,
    ):
        self.clock = Clock(0)
        self.dt = dt

        builder = SimpleTreeBuilder(self.clock)
        builder.sensor_disturbances = [GaussianTimeSampler(0.0, 1.0, 5.0, 100.0) for _ in range(num_sensors)]
        builder.sensor_epochs = [0 for _ in range(num_sensors)]
        builder.sensor_periods = [100 for _ in range(num_sensors)]
        builder.compute_duration = GammaDistributionSampler(5.0, 1.0, 3.0, 70.0)
        builder.age_normalizer = ConstantNormalizer(100.0)
        builder.build()
        self.system = builder.system

        self._sinks = [builder.nodes["LOST_BUFFER"], builder.nodes["LOST_COMPUTE"]]
        self._output = builder.nodes["OUTPUT"]

        self.action_space = gym.spaces.Discrete(int((1 - 2**self.system.num_action) / (1 - 2)))
        lb = -np.inf * np.ones((len(self.system.state),), dtype=float)
        ub = +np.inf * np.ones((len(self.system.state),), dtype=float)
        self.observation_space = gym.spaces.Box(lb, ub, dtype=float)

        self.cost_message_loss = cost_message_loss
        self.cost_output_time = cost_output_time
        self.cost_input = cost_input

    def reset(self, seed=None):
        super().reset(seed=seed)
        np.random.seed(seed)
        self.clock.reset()
        self.system.reset()
        self.system.update()
        return np.array(self.system.state), {}

    def unpack_action(self, action: int) -> np.ndarray:
        assert action >= 0 and action <= 255
        bits = np.unpackbits(np.uint8(action))
        return bits[-self._system.num_action_nodes :]

    def step(self, action: int):
        # Reset sinks; lost messages are counted from now on
        for sink in self._sinks:
            sink.reset()

        # Propagate system
        action = self.unpack_action()
        self.system.act(action)
        self.clock += self.dt
        self.system.update()

        # Get state and reward
        state = np.array(self.system.state)
        reward = -self.cost_message_loss * float(self._count_lost_msgs())
        if self._output.last_received:
            age = as_age(self._output.last_received.header.t_measure_oldest, self.clock.get_time())
            reward -= self.cost_output_time * float(age)
        reward -= self.cost_input * np.sum(action)

        return state, reward, False, False, {}

    def _count_lost_msgs(self) -> int:
        n = 0
        for sink in self._sinks:
            n += len(sink.received_messages())
        return n