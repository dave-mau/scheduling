import gymnasium as gym
import numpy as np
from computation_sim.basic_types import Time
from computation_sim.example_systems import SimpleTreeBuilder
from computation_sim.nodes import ConstantNormalizer
from computation_sim.system import num_actions, unpack_action
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
        filter_threshold=np.inf,
    ):
        # Store init params
        self.clock = Clock(0)
        self.dt = dt
        self.cost_message_loss = cost_message_loss
        self.cost_output_time = cost_output_time
        self.cost_input = cost_input

        # Build system
        self.age_normalizer = ConstantNormalizer(100.0)
        builder = SimpleTreeBuilder(self.clock)
        builder.sensor_disturbances = [GaussianTimeSampler(0.0, 1.0, 5.0, 100.0) for _ in range(num_sensors)]
        builder.sensor_epochs = [0 for _ in range(num_sensors)]
        builder.sensor_periods = [100 for _ in range(num_sensors)]
        builder.compute_duration = GammaDistributionSampler(5.0, 1.0, 2.5, 70.0)
        builder.age_normalizer = self.age_normalizer
        builder.filter_threshold = filter_threshold
        builder.build()
        self.system = builder.system

        # Init system
        self.system.update()
        self._sinks = [builder.nodes["LOST_BUFFER"], builder.nodes["LOST_COMPUTE"]]
        self._output = builder.nodes["OUTPUT"]

        # Keep nodes as an easy entry point for experiments
        self.nodes = builder.nodes

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
        for sink in self._sinks:
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
        return sum(len(sink.received_messages) for sink in self._sinks)
