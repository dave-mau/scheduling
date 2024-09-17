from typing import List

from computation_sim.basic_types import Time
from computation_sim.nodes import (
    FilteringMISONode,
    Node,
    OutputNode,
    PeriodicEpochSensor,
    RingBufferNode,
    SinkNode,
    SourceNode,
    StateVariableNormalizer,
)
from computation_sim.system import Action, System, SystemBuidler
from computation_sim.time import Clock, DurationSampler
from numpy import inf


class SimpleTreeBuilder(SystemBuidler):
    def __init__(
        self,
        clock: Clock,
        sensor_epochs: List[Time] = 0,
        sensor_periods: List[Time] = 0,
        sensor_disturbances: List[DurationSampler] = None,
        compute_duration: DurationSampler = None,
        age_normalizer: StateVariableNormalizer = None,
        occupancy_normalizer: StateVariableNormalizer = None,
        count_normalizer: StateVariableNormalizer = None,
        filter_threshold: float = inf,
    ):
        self.clock = clock
        self._system = None
        self._nodes = {}
        self.sensor_epochs = sensor_epochs
        self.sensor_periods = sensor_periods
        self.sensor_disturbances = sensor_disturbances
        self.compute_duration = compute_duration
        self.age_normalizer = age_normalizer
        self.occupancy_normalizer = occupancy_normalizer
        self.count_normalizer = count_normalizer
        self.filter_threshold = filter_threshold

    def build(self) -> None:
        assert len(self.sensor_epochs) == len(self.sensor_periods)
        assert len(self.sensor_periods) == len(self.sensor_disturbances)

        self._nodes["OUTPUT"] = OutputNode(
            self.clock.as_readonly(),
            id="OUTPUT",
            age_normalizer=self.age_normalizer,
            occupancy_normalizer=self.occupancy_normalizer,
        )
        self._nodes["LOST_BUFFER"] = SinkNode(
            self.clock.as_readonly(), id="LOST_BUFFER", count_normalizer=self.count_normalizer
        )
        self._nodes["LOST_COMPUTE"] = SinkNode(
            self.clock.as_readonly(), id="LOST_COMPUTE", count_normalizer=self.count_normalizer
        )
        self._nodes["COMPUTE"] = FilteringMISONode(
            self.clock.as_readonly(),
            self.compute_duration,
            id="COMPUTE",
            age_normalizer=self.age_normalizer,
            occupancy_normalizer=self.occupancy_normalizer,
            filter_threshold=self.filter_threshold,
        )
        self._nodes["COMPUTE"].set_output_pass(self._nodes["OUTPUT"])
        self._nodes["COMPUTE"].set_output_fail(self._nodes["LOST_COMPUTE"])

        compute_action = Action(name="ACT_COMPUTE_NODE")
        for i, (epoch, period, dist) in enumerate(
            zip(self.sensor_epochs, self.sensor_periods, self.sensor_disturbances)
        ):
            sensor = PeriodicEpochSensor(epoch, period, dist)
            self._nodes[f"SOURCE_{i}"] = SourceNode(self.clock.as_readonly(), sensor, f"SOURCE_{i}")
            self._nodes[f"BUFFER_{i}"] = RingBufferNode(
                self.clock.as_readonly(),
                f"BUFFER_{i}",
                max_num_elements=1,
                age_normalizer=self.age_normalizer,
                occupancy_normalizer=self.occupancy_normalizer,
            )
            self._nodes[f"SOURCE_{i}"].add_output(self._nodes[f"BUFFER_{i}"])
            self._nodes[f"BUFFER_{i}"].set_output(self._nodes["COMPUTE"])
            self._nodes[f"BUFFER_{i}"].set_overflow_output(self._nodes["LOST_BUFFER"])
            compute_action.register_callback(self._nodes[f"BUFFER_{i}"].trigger, 1)
        compute_action.register_callback(self._nodes["COMPUTE"].trigger, 0)
        compute_action.register_readiness_callback(lambda: not self._nodes["COMPUTE"].is_busy)

        self._system = System()
        self._system.add_action(compute_action)
        for node in self._nodes.values():
            self._system.add_node(node)
