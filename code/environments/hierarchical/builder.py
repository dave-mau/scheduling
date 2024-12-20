from typing import List

import numpy as np
from computation_sim.basic_types import Time
from computation_sim.nodes import (
    FilteringMISONode,
    OutputNode,
    PeriodicEpochSensor,
    RingBufferNode,
    SinkNode,
    SourceNode,
    StateVariableNormalizer,
)
from computation_sim.system import Action, System, SystemBuidler
from computation_sim.time import Clock, DurationSampler

from .types import ActionCollection, SystemCollection


class HierarchicalSystemBuilder(SystemBuidler):
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

        self._system: System = None
        self._sources: List[SourceNode] = []
        self._sinks: List[SourceNode] = []
        self._action_collections: List[ActionCollection] = []
        self._output: OutputNode = None
        self._nodes = {}
        self._samplers = []

        self._init_sinks()

    @property
    def system_collection(self) -> SystemCollection:
        return SystemCollection(
            system=self._system,
            sources=self._sources,
            sinks=self._sinks,
            action_collections=self._action_collections,
            output=self._output,
            samplers=self._samplers,
        )

    def _init_sinks(self):
        sinks = dict(
            SENS_BUF_LOST=SinkNode(self.clock.as_readonly(), id="SENS_BUF_LOST"),
            SENS_CMP_LOST=SinkNode(self.clock.as_readonly(), id="SENS_CMP_LOST"),
            SENS_CMP_BUF_LOST=SinkNode(self.clock.as_readonly(), id="SENS_CMP_BUF_LOST"),
            EDGE_CMP_LOST=SinkNode(self.clock.as_readonly(), id="EDGE_CMP_LOST"),
            EDGE_CMP_BUF_LOST=SinkNode(self.clock.as_readonly(), id="EDGE_CMP_BUF_LOST"),
            OUTPUT_CMP_LOST=SinkNode(self.clock.as_readonly(), id="OUTPUT_CMP_LOST"),
        )
        self._nodes.update(sinks)
        self._sinks = [
            sinks["SENS_BUF_LOST"],
            sinks["SENS_CMP_BUF_LOST"],
            sinks["EDGE_CMP_BUF_LOST"],
        ]

    def add_sensor_chain(
        self,
        id: str,
        sensor_epoch: Time,
        sensor_period: Time,
        sensor_disturbance: DurationSampler,
        compute_duration: DurationSampler,
    ) -> RingBufferNode:
        # Update samplers
        self._samplers.append(sensor_disturbance)
        self._samplers.append(compute_duration)

        # Sensor
        sensor = PeriodicEpochSensor(sensor_epoch, sensor_period, sensor_disturbance)
        source_node = SourceNode(self.clock.as_readonly(), sensor, f"SENS_{id}")
        self._nodes[source_node.id] = source_node
        self._sources.append(source_node)

        # Sensor Output Buffer
        sensor_buffer_node = RingBufferNode(
            self.clock.as_readonly(),
            id=f"SENS_BUF_{id}",
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
            id=f"SENS_CMP_{id}",
            age_normalizer=self.age_normalizer,
            occupancy_normalizer=self.occupancy_normalizer,
            receive_cb=lambda node: node.trigger(),
        )
        self._nodes[compute_node.id] = compute_node

        # Compute output buffer
        compute_buffer_node = RingBufferNode(
            self.clock.as_readonly(),
            id=f"SENS_CMP_BUF_{id}",
            max_num_elements=1,
            age_normalizer=self.age_normalizer,
            occupancy_normalizer=self.occupancy_normalizer,
        )
        self._nodes[compute_buffer_node.id] = compute_buffer_node

        # Connect the nodes
        source_node.add_output(sensor_buffer_node)
        sensor_buffer_node.set_output(compute_node)
        sensor_buffer_node.set_overflow_output(self._nodes["SENS_BUF_LOST"])
        compute_node.set_output_pass(compute_buffer_node)
        compute_node.set_output_fail(self._nodes["SENS_CMP_LOST"])
        compute_buffer_node.set_overflow_output(self._nodes["SENS_CMP_BUF_LOST"])
        return compute_buffer_node

    def add_edge_compute(
        self,
        id: str,
        inputs: List[RingBufferNode],
        compute_duration: DurationSampler,
        filter_threshold: float = np.inf,
    ) -> RingBufferNode:
        # Update sampler
        self._samplers.append(compute_duration)

        # Edge Compute Node
        compute_node = FilteringMISONode(
            self.clock.as_readonly(),
            duration_sampler=compute_duration,
            id=f"EDGE_CMP_{id}",
            filter_threshold=filter_threshold,
            age_normalizer=self.age_normalizer,
            occupancy_normalizer=self.occupancy_normalizer,
        )
        self._nodes[compute_node.id] = compute_node

        # Add a buffer for the output
        buffer_node = RingBufferNode(
            self.clock.as_readonly(),
            id=f"EDGE_CMP_BUF_{id}",
            max_num_elements=1,
            age_normalizer=self.age_normalizer,
            occupancy_normalizer=self.occupancy_normalizer,
        )
        self._nodes[buffer_node.id] = buffer_node

        # Add an action and connect the nodes
        action = Action(f"EDGE_CMP_{id}_ACT")
        for input in inputs:
            input.set_output(compute_node)
            action.register_callback(input.trigger, 1)
        action.register_callback(compute_node.trigger, 0)

        # Action can only be executed if the compute node is not busy
        action.register_readiness_callback(lambda: not compute_node.is_busy)
        self._action_collections.append(ActionCollection(inputs, compute_node, action))

        compute_node.set_output_pass(buffer_node)
        compute_node.set_output_fail(self._nodes["EDGE_CMP_LOST"])
        buffer_node.set_overflow_output(self._nodes["EDGE_CMP_BUF_LOST"])
        return buffer_node

    def add_output_compute(
        self,
        inputs: List[RingBufferNode],
        compute_duration: DurationSampler,
        filter_threshold: float = np.inf,
    ):
        # Update sampler
        self._samplers.append(compute_duration)

        # Output Compute Node
        compute_node = FilteringMISONode(
            self.clock.as_readonly(),
            duration_sampler=compute_duration,
            id="OUTPUT_CMP",
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
        compute_node.set_output_fail(self._nodes["OUTPUT_CMP_LOST"])

        # Setup the action
        action = Action("OUTPUT_CMP_ACT")
        for input in inputs:
            input.set_output(compute_node)
            action.register_callback(input.trigger, 1)
        action.register_callback(compute_node.trigger, 0)
        action.register_readiness_callback(lambda: not compute_node.is_busy)
        self._action_collections.append(ActionCollection(inputs, compute_node, action))
        self._output = output_node

    def build(self) -> None:
        # Build the system
        self._system = System()
        for action_collection in self._action_collections:
            self._system.add_action(action_collection.action)
        for node in self._nodes.values():
            self._system.add_node(node)

        # Update the system once, to force-set the update list and node graph
        self._system.update()
