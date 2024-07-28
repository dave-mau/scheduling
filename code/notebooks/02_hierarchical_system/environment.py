import gymnasium as gym
import numpy as np
from computation_sim.basic_types import Time
from computation_sim.nodes import ConstantNormalizer
from computation_sim.system import num_actions, unpack_action
from computation_sim.time import (
    Clock,
    GammaDistributionSampler,
    GaussianTimeSampler,
    as_age,
)
from system import HierarchicalSystemBuilder


class MultiStageEnv(gym.Env):
    def __init__(
        self,
        dt: Time = 10,
        cost_message_loss=1.0,
        cost_output_time=0.1,
        cost_input=0.01,
    ):
        # Store init params
        self.clock = Clock(0)
        self.dt = dt
        self.cost_message_loss = cost_message_loss
        self.cost_output_time = cost_output_time
        self.cost_input = cost_input

        # Build system
        self.age_normalizer = ConstantNormalizer(100.0)
        self.count_normalizer = ConstantNormalizer(1.0)
        self.occupancy_normalizer = ConstantNormalizer(1.0)
        builder = HierarchicalSystemBuilder(
            self.clock, self.age_normalizer, self.count_normalizer, self.occupancy_normalizer
        )

        # Set-up the sensor chains
        e0 = []
        e0.append(
            builder.add_sensor_chain(
                "0", 0, 100, GaussianTimeSampler(0.0, 1.0, 5.0, 100.0), GammaDistributionSampler(5.0, 1.0, 2.5, 50.0)
            )
        )
        e0.append(
            builder.add_sensor_chain(
                "1", 0, 100, GaussianTimeSampler(0.0, 1.0, 5.0, 100.0), GammaDistributionSampler(5.0, 1.0, 2.5, 50.0)
            )
        )
        e0.append(
            builder.add_sensor_chain(
                "2", 0, 100, GaussianTimeSampler(0.0, 1.0, 5.0, 100.0), GammaDistributionSampler(5.0, 1.0, 2.5, 50.0)
            )
        )

        e1 = []
        e1.append(
            builder.add_sensor_chain(
                "3", 0, 100, GaussianTimeSampler(0.0, 1.0, 5.0, 100.0), GammaDistributionSampler(5.0, 1.0, 2.5, 50.0)
            )
        )
        e1.append(
            builder.add_sensor_chain(
                "4", 0, 100, GaussianTimeSampler(0.0, 1.0, 5.0, 100.0), GammaDistributionSampler(5.0, 1.0, 2.5, 50.0)
            )
        )
        e1.append(
            builder.add_sensor_chain(
                "5", 0, 100, GaussianTimeSampler(0.0, 1.0, 5.0, 100.0), GammaDistributionSampler(5.0, 1.0, 2.5, 50.0)
            )
        )

        e2 = []
        e2.append(
            builder.add_sensor_chain(
                "6", 0, 100, GaussianTimeSampler(0.0, 1.0, 5.0, 100.0), GammaDistributionSampler(5.0, 1.0, 2.5, 50.0)
            )
        )
        e2.append(
            builder.add_sensor_chain(
                "7", 0, 100, GaussianTimeSampler(0.0, 1.0, 5.0, 100.0), GammaDistributionSampler(5.0, 1.0, 2.5, 50.0)
            )
        )
        e2.append(
            builder.add_sensor_chain(
                "8", 0, 100, GaussianTimeSampler(0.0, 1.0, 5.0, 100.0), GammaDistributionSampler(5.0, 1.0, 2.5, 50.0)
            )
        )

        e3 = []
        e3.append(
            builder.add_sensor_chain(
                "9", 0, 100, GaussianTimeSampler(0.0, 1.0, 5.0, 100.0), GammaDistributionSampler(5.0, 1.0, 2.5, 50.0)
            )
        )
        e3.append(
            builder.add_sensor_chain(
                "10", 0, 100, GaussianTimeSampler(0.0, 1.0, 5.0, 100.0), GammaDistributionSampler(5.0, 1.0, 2.5, 50.0)
            )
        )

        # Set-up the edge nodes
        m = []
        m.append(builder.add_edge_compute("0", e0, GammaDistributionSampler(5.0, 1.0, 2.5, 50.0), 100.0))
        m.append(builder.add_edge_compute("1", e1, GammaDistributionSampler(5.0, 1.0, 2.5, 50.0), 100.0))
        m.append(builder.add_edge_compute("2", e2, GammaDistributionSampler(5.0, 1.0, 2.5, 50.0), 100.0))
        m.append(builder.add_edge_compute("3", e3, GammaDistributionSampler(5.0, 1.0, 2.5, 50.0), 100.0))

        # Set-up the output node
        builder.add_output_compute(m, GammaDistributionSampler(5.0, 1.0, 2.5, 60.0), 100.0)
        builder.build()

        self.system = builder.system
        self.system.update()
        self._sinks = [
            builder._nodes["SENSOR_BUFFER_LOST"],
            builder._nodes["SENSOR_COMPUTE_LOST"],
            builder._nodes["SENSOR_COMPUTE_BUFFER_LOST"],
            builder._nodes["EDGE_COMPUTE_LOST"],
            builder._nodes["EDGE_COMPUTE_BUFFER_LOST"],
            builder._nodes["OUTPUT_COMPUTE_LOST"],
        ]
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
