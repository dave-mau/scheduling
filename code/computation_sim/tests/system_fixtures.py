from typing import Tuple

import pytest
from computation_sim.nodes import (
    ConstantNormalizer,
    FilteringMISONode,
    Node,
    OutputNode,
    PeriodicEpochSensor,
    RingBufferNode,
    SinkNode,
    SourceNode,
)
from computation_sim.system import Action, System
from computation_sim.time import Clock, FixedDuration, TimeProvider


@pytest.fixture
def simple_chain() -> Tuple[System, Clock, Tuple[Node]]:
    clock = Clock(0)

    # Setup the source
    sensor = PeriodicEpochSensor(0, 100, FixedDuration(0))
    source_node = SourceNode(clock.as_readonly(), sensor, "SOURCE_NODE")
    # Setup the processor node
    compute_node = FilteringMISONode(clock.as_readonly(), FixedDuration(10), id="COMPUTE_NODE")
    # Setup the output node
    output_node = OutputNode(clock.as_readonly(), id="OUTPUT_NODE")
    # Setup a sink for failing compute
    sink_node = SinkNode(clock.as_readonly(), id="SINK_NODE")

    # Connect outputs
    source_node.add_output(compute_node)
    compute_node.set_output_pass(output_node)
    compute_node.set_output_fail(sink_node)

    # Set-up the action
    compute_action = Action(name="ACT_COMPUTE_NODE")
    compute_action.register_callback(compute_node.trigger, 0)

    system = System()
    system.add_action(compute_action)
    system.add_node(sink_node)
    system.add_node(output_node)
    system.add_node(compute_node)
    system.add_node(source_node)
    return system, clock, (source_node, compute_node, output_node, sink_node)


@pytest.fixture
def simple_tree() -> Tuple[System, Clock, Tuple[Node]]:
    clock = Clock(0)

    # Normalizer for time
    normalizer = ConstantNormalizer(100.0)

    # Setup the source
    sensor = PeriodicEpochSensor(0, 100, FixedDuration(0))
    source_node_0 = SourceNode(clock.as_readonly(), sensor, "SOURCE_NODE_0")
    sensor = PeriodicEpochSensor(0, 100, FixedDuration(0))
    source_node_1 = SourceNode(clock.as_readonly(), sensor, "SOURCE_NODE_1")

    # Setup sink and output
    output_node = OutputNode(clock.as_readonly(), id="OUTPUT_NODE", age_normalizer=normalizer)
    lost_node = SinkNode(clock.as_readonly(), id="LOST_MESSAGES")

    # Setup buffers
    buffer_0 = RingBufferNode(
        clock.as_readonly(),
        "BUFFER_0",
        overflow_cb=lambda x: lost_node.receive(x),
        age_normalizer=normalizer,
    )
    buffer_1 = RingBufferNode(
        clock.as_readonly(),
        "BUFFER_1",
        overflow_cb=lambda x: lost_node.receive(x),
        age_normalizer=normalizer,
    )

    # Setup the processor node
    compute_node = FilteringMISONode(
        clock.as_readonly(),
        FixedDuration(10),
        id="COMPUTE_NODE",
        age_normalizer=normalizer,
    )

    # Connect outputs
    source_node_0.add_output(buffer_0)
    source_node_1.add_output(buffer_1)
    buffer_0.add_output(compute_node)
    buffer_1.add_output(compute_node)
    compute_node.set_output_fail(lost_node)
    compute_node.set_output_pass(output_node)

    # Set-up the action
    compute_action = Action(name="ACT_COMPUTE_NODE")
    compute_action.register_callback(buffer_0.trigger, 1)
    compute_action.register_callback(buffer_1.trigger, 1)
    compute_action.register_callback(compute_node.trigger, 0)

    system = System()
    system.add_action(compute_action)
    system.add_node(lost_node)
    system.add_node(output_node)
    system.add_node(compute_node)
    system.add_node(source_node_0)
    system.add_node(source_node_1)
    system.add_node(buffer_0)
    system.add_node(buffer_1)
    return (
        system,
        clock,
        (
            source_node_0,
            source_node_1,
            buffer_0,
            buffer_1,
            compute_node,
            output_node,
            lost_node,
        ),
    )
