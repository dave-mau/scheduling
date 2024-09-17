from typing import List

import numpy as np
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


class ThreeStageSystemBuilder(SystemBuidler):
    def __init__(
        self,
        clock: Clock,
        age_normalizer: StateVariableNormalizer = None,
        count_normalizer: StateVariableNormalizer = None,
        occupancy_normalizer: StateVariableNormalizer = None,
    ):
        self.clock = clock
        self.age_normalizer = age_normalizer
        self.count_normalizer = count_normalizer
        self.occupancy_normalizer = occupancy_normalizer
        self._system = None
        self._nodes = {}
        self._actions = []

        self._init_sinks()

    def _init_sinks(self):
        # Initialize SinkNodes for verflow and fail outputs
        self._nodes["SENSOR_BUFFER_LOST"] = SinkNode(self.clock.as_readonly(), id="SENSOR_BUFFER_LOST")
        self._nodes["SENSOR_COMPUTE_LOST"] = SinkNode(self.clock.as_readonly(), id="SENSOR_COMPUTE_LOST")
        self._nodes["SENSOR_COMPUTE_BUFFER_LOST"] = SinkNode(self.clock.as_readonly(), id="SENSOR_COMPUTE_BUFFER_LOST")
        self._nodes["EDGE_COMPUTE_LOST"] = SinkNode(self.clock.as_readonly(), id="EDGE_COMPUTE_LOST")
        self._nodes["EDGE_COMPUTE_BUFFER_LOST"] = SinkNode(self.clock.as_readonly(), id="EDGE_COMPUTE_BUFFER_LOST")
        self._nodes["OUTPUT_COMPUTE_LOST"] = SinkNode(self.clock.as_readonly(), id="OUTPUT_COMPUTE_LOST")

    def add_sensor_chain(
        self,
        id: str,
        sensor_epoch: Time,
        sensor_period: Time,
        sensor_disturbance: DurationSampler,
        compute_duration: DurationSampler,
    ) -> RingBufferNode:

        # Sensor
        sensor = PeriodicEpochSensor(sensor_epoch, sensor_period, sensor_disturbance)
        source_node = SourceNode(self.clock.as_readonly(), sensor, f"SENSOR_{id}")
        self._nodes[source_node.id] = source_node

        # Sensor Output Buffer
        sensor_buffer_node = RingBufferNode(
            self.clock.as_readonly(),
            id=f"SENSOR_BUFFER_{id}",
            max_num_elements=1,
            age_normalizer=self.age_normalizer,
            occupancy_normalizer=self.occupancy_normalizer,
        )
        sensor_buffer_node.set_receive_cb(lambda node: node.trigger())
        self._nodes[sensor_buffer_node.id] = sensor_buffer_node

        # Sensor Compute Node
        compute_node = FilteringMISONode(
            self.clock.as_readonly(),
            duration_sampler=compute_duration,
            id=f"SENSOR_COMPUTE_{id}",
            age_normalizer=self.age_normalizer,
            occupancy_normalizer=self.occupancy_normalizer,
            receive_cb=lambda node: node.trigger(),
        )
        self._nodes[compute_node.id] = compute_node

        # Compute output buffer
        compute_buffer_node = RingBufferNode(
            self.clock.as_readonly(),
            id=f"SENSOR_COMPUTE_BUFFER_{id}",
            max_num_elements=1,
            age_normalizer=self.age_normalizer,
            occupancy_normalizer=self.occupancy_normalizer,
        )
        self._nodes[compute_buffer_node.id] = compute_buffer_node

        # Connect the nodes
        source_node.add_output(sensor_buffer_node)
        sensor_buffer_node.set_output(compute_node)
        sensor_buffer_node.set_overflow_output(self._nodes["SENSOR_BUFFER_LOST"])
        compute_node.set_output_pass(compute_buffer_node)
        compute_node.set_output_fail(self._nodes["SENSOR_COMPUTE_LOST"])
        compute_buffer_node.set_overflow_output(self._nodes["SENSOR_COMPUTE_BUFFER_LOST"])
        return compute_buffer_node

    def add_edge_compute(
        self,
        id: str,
        inputs: List[RingBufferNode],
        compute_duration: DurationSampler,
        filter_threshold: float = np.inf,
    ) -> RingBufferNode:

        # Edge Compute Node
        compute_node = FilteringMISONode(
            self.clock.as_readonly(),
            duration_sampler=compute_duration,
            id=f"EDGE_COMPUTE_{id}",
            filter_threshold=filter_threshold,
            age_normalizer=self.age_normalizer,
            occupancy_normalizer=self.occupancy_normalizer,
        )
        self._nodes[compute_node.id] = compute_node

        # Add a buffer for the output
        buffer_node = RingBufferNode(
            self.clock.as_readonly(),
            id=f"EDGE_COMPUTE_BUFFER_{id}",
            max_num_elements=1,
            age_normalizer=self.age_normalizer,
            occupancy_normalizer=self.occupancy_normalizer,
        )
        self._nodes[buffer_node.id] = buffer_node

        # Add an action and connect the nodes
        action = Action(f"EDGE_COPMPUTE_{id}_ACT")
        for input in inputs:
            input.set_output(compute_node)
            action.register_callback(input.trigger, 1)
        action.register_callback(compute_node.trigger, 0)

        # Action can only be executed if the compute node is not busy
        action.register_readiness_callback(lambda: not compute_node.is_busy)
        self._actions.append(action)

        compute_node.set_output_pass(buffer_node)
        compute_node.set_output_fail(self._nodes["EDGE_COMPUTE_LOST"])
        buffer_node.set_overflow_output(self._nodes["EDGE_COMPUTE_BUFFER_LOST"])
        return buffer_node

    def add_output_compute(
        self, inputs: List[RingBufferNode], compute_duration: DurationSampler, filter_threshold: float = np.inf
    ):
        # Output Compute Node
        compute_node = FilteringMISONode(
            self.clock.as_readonly(),
            duration_sampler=compute_duration,
            id="OUTPUT_COMPUTE",
            filter_threshold=filter_threshold,
            age_normalizer=self.age_normalizer,
            occupancy_normalizer=self.occupancy_normalizer,
        )
        self._nodes[compute_node.id] = compute_node

        # Output node
        output_node = OutputNode(
            self.clock.as_readonly(),
            id="OUTPUT",
            age_normalizer=self.age_normalizer,
            occupancy_normalizer=self.occupancy_normalizer,
        )
        self._nodes[output_node.id] = output_node

        # Connect the nodes
        compute_node.set_output_pass(output_node)
        compute_node.set_output_fail(self._nodes["OUTPUT_COMPUTE_LOST"])

        # Setup the action
        action = Action("OUTPUT_COMPUTE_ACT")
        for input in inputs:
            input.set_output(compute_node)
            action.register_callback(input.trigger, 1)
        action.register_callback(compute_node.trigger, 0)
        action.register_readiness_callback(lambda: not compute_node.is_busy)
        self._actions.append(action)

    def build(self) -> None:
        # Build the system
        self._system = System()
        for action in self._actions:
            self._system.add_action(action)
        for node in self._nodes.values():
            self._system.add_node(node)
