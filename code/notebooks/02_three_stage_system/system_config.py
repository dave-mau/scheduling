import sys
sys.path.insert(0, "/home/davidmauderli/repos/scheduling/code/")

from computation_sim.nodes import ConstantNormalizer
from computation_sim.time import Clock, GammaDistributionSampler, GaussianTimeSampler
from environments.hierarchical import (
    HierarchicalSystemBuilder,
    Reward,
)
class SystemConfig:
    def __init__(self, dt=10):
        self.input_sampler = GaussianTimeSampler(0.0, 1.0, 1.0, 100.0)
        self.input_compute_sampler = GammaDistributionSampler(5.0, 1.0, 3.0, 30.0)
        self.edge_compute_sampler = GammaDistributionSampler(3.0, 1.0, 1.0, 30.0)
        self.global_compute_sampler = GammaDistributionSampler(9.0, 1.0, 3.0, 30.0)
        self.age_normalizer = ConstantNormalizer(100.0)
        self.count_normalizer = ConstantNormalizer(1.0)
        self.occupancy_normalizer = ConstantNormalizer(1.0)
        self.dt = dt
        self.cost_message_loss = 1.0
        self.cost_output_time = 0.1 / 100.0
        self.cost_input = 0.01

    def make(self) -> dict:
        clock = Clock(0)
        builder = HierarchicalSystemBuilder(
            clock, self.age_normalizer, self.count_normalizer, self.occupancy_normalizer
        )

        # Set-up the sensor chains
        s0 = [
            builder.add_sensor_chain("0", 0, 100, self.input_sampler, self.input_compute_sampler),
            builder.add_sensor_chain("1", 0, 100, self.input_sampler, self.input_compute_sampler),
        ]
        s1 = [
            builder.add_sensor_chain("2", 0, 100, self.input_sampler, self.input_compute_sampler),
            builder.add_sensor_chain("3", 0, 100, self.input_sampler, self.input_compute_sampler),
        ]

        # Set-up the edge nodes
        m = [
            builder.add_edge_compute("0", s0, self.edge_compute_sampler, 90.0),
            builder.add_edge_compute("1", s1, self.edge_compute_sampler, 90.0),
        ]

        # Set-up the output node
        builder.add_output_compute(m, self.global_compute_sampler, 90.0)
        builder.build()

        # Build the reward
        reward = Reward(self.cost_message_loss, self.cost_output_time, self.cost_input)

        return dict(clock=clock, system_collection=builder.system_collection, reward=reward, dt=self.dt)
