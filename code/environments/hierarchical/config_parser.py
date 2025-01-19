from typing import Optional
from collections import defaultdict

from computation_sim.nodes import (
    ConstantNormalizer,
    RingBufferNode,
    StateVariableNormalizer,
)
from computation_sim.time import (
    Clock,
    DurationSampler,
    FixedDuration,
    GammaDistributionSampler,
    GaussianTimeSampler,
)

from .builder import HierarchicalSystemBuilder
from .reward import Reward


class ConfigParser:
    def __init__(self):
        pass

    def parse(self, config: dict) -> Optional[dict]:
        clock = self.parse_clock(config["clock"])
        age_normalizer = self.parse_normalizer(config["age_normalizer"])
        count_normalizer = self.parse_normalizer(config["count_normalizer"])
        occupancy_normalizer = self.parse_normalizer(config["occupancy_normalizer"])
        reward = self.parse_reward(config["reward"])

        builder = HierarchicalSystemBuilder(clock, age_normalizer, count_normalizer, occupancy_normalizer)
        self.parse_nodes(config, builder)

        return dict(clock=clock, system_collection=builder.system_collection, reward=reward, dt=config["dt"])

    def parse_clock(self, config: dict) -> Clock:
        return Clock(config["initial_time"])

    def parse_normalizer(self, config: dict) -> StateVariableNormalizer:
        if config["type"] == "ConstantNormalizer":
            return ConstantNormalizer(config["value"])
        else:
            raise ValueError(f"Unknown age normalizer type: {config['type']}")

    def parse_nodes(self, config: dict, builder: HierarchicalSystemBuilder):
        #default dict where each entry is an empty list
        edge_inputs = defaultdict(list)
        for c in config["sensors"]:
            edge_inputs[c["next_id"]].append(
                builder.add_sensor_chain(
                    c["id"],
                    c["epoch"],
                    c["period"],
                    self.parse_sampler(c["sensor_sampler"]),
                    self.parse_sampler(c["compute_sampler"]),
                ))

        for c in config["edge_computes"]:
            edge_inputs[c["next_id"]].append(
                builder.add_edge_compute(
                    c["id"],
                    edge_inputs[c["id"]],
                    self.parse_sampler(c["compute_sampler"]),
                    c["filter_threshold"],
                ))

        builder.add_output_compute(
            edge_inputs["output"]["id"],
            self.parse_sampler(config["output"]["compute_sampler"]),
            config["output"]["filter_threshold"],
        )
        builder.build()

    def parse_reward(self, config: dict) -> Reward:
        return Reward(
            config["cost_message_loss"],
            config["cost_output_time"],
            config["cost_input"])


    def parse_sensor_chain(self, config: dict, builder: HierarchicalSystemBuilder) -> RingBufferNode:
        return

    def parse_sampler(self, config: dict) -> DurationSampler:
        if config["type"] == "GaussianTimeSampler":
            return GaussianTimeSampler(config["mu"], config["std"], config["gain"], config["offset"])
        elif config["type"] == "GammaDistributionSampler":
            return GammaDistributionSampler(config["k"], config["theta"], config["gain"], config["offset"])
        elif config["type"] == "FixedDuration":
            return FixedDuration(config["val"])
        else:
            raise ValueError(f"Unknown sampler type: {config['type']}")
