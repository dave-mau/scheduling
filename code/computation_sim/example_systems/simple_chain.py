from computation_sim.basic_types import Time
from computation_sim.nodes import (
    FilteringMISONode,
    OutputNode,
    PeriodicEpochSensor,
    SinkNode,
    SourceNode,
    StateVariableNormalizer,
)
from computation_sim.system import Action, System, SystemBuidler
from computation_sim.time import Clock, DurationSampler


class SimpleChainBuilder(SystemBuidler):
    def __init__(
        self,
        clock: Clock,
        sensor_epoch: Time = 0,
        sensor_period: Time = 0,
        sensor_disturbance: DurationSampler = None,
        compute_duration: DurationSampler = None,
        age_normalizer: StateVariableNormalizer = None,
        occupancy_normalizer: StateVariableNormalizer = None,
        count_normalizer: StateVariableNormalizer = None,
    ):
        self.clock = clock
        self._system = None
        self._nodes = {}
        self.sensor_epoch = sensor_epoch
        self.sensor_period = sensor_period
        self.sensor_disturbance = sensor_disturbance
        self.compute_duration = compute_duration
        self.age_normalizer = age_normalizer
        self.occupancy_normalizer = occupancy_normalizer
        self.count_normalizer = count_normalizer

    def build(self) -> None:
        # Setup the nodes
        # 1. SOURCE
        # sensor_config = self.config["source"]["sensor"]
        sensor = PeriodicEpochSensor(self.sensor_epoch, self.sensor_period, self.sensor_disturbance)
        self._nodes["SOURCE"] = SourceNode(self.clock.as_readonly(), sensor, id="SOURCE")

        # 2. COMPUTE
        self._nodes["COMPUTE"] = FilteringMISONode(
            self.clock.as_readonly(),
            self.compute_duration,
            id="COMPUTE",
            occupancy_normalizer=self.occupancy_normalizer,
            age_normalizer=self.age_normalizer,
        )

        # 3. OUTPUT
        self._nodes["OUTPUT"] = OutputNode(
            self.clock.as_readonly(),
            id="OUTPUT",
            age_normalizer=self.age_normalizer,
            occupancy_normalizer=self.occupancy_normalizer,
        )

        # 4. SINK
        self._nodes["SINK"] = SinkNode(
            self.clock.as_readonly(), id="SINK_NODE", count_normalizer=self.count_normalizer
        )

        # Connect outputs
        self._nodes["SOURCE"].add_output(self._nodes["COMPUTE"])
        self._nodes["COMPUTE"].set_output_pass(self._nodes["OUTPUT"])
        self._nodes["COMPUTE"].set_output_fail(self._nodes["SINK"])

        # Set-up the action
        compute_action = Action(name="ACT_COMPUTE")
        compute_action.register_callback(self._nodes["COMPUTE"].trigger, 0)

        self._system = System()
        self._system.add_action(compute_action)
        for node in self._nodes.values():
            self._system.add_node(node)
