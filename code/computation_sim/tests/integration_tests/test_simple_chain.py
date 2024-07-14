from typing import Tuple

import numpy as np
import pytest
from computation_sim.nodes import (
    FilteringMISONode,
    Node,
    OutputNode,
    PeriodicEpochSensor,
    SinkNode,
    SourceNode,
)
from computation_sim.system import Action, System
from computation_sim.time import Clock, FixedDuration, TimeProvider


@pytest.fixture
def setup() -> Tuple[System, Clock, Tuple[Node]]:
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


def test_no_action_no_output(setup):
    system, clock, nodes = setup
    while clock.get_time() < 1000:
        clock += 1
        system.update()

    assert nodes[2].last_received is None


def test_nominal(setup):
    system, clock, nodes = setup
    # state vector:
    # [compute_busy, clock-time, output has result, ..output-times.., sink-count]

    # t: [0, 50) -- Wait and do nothing
    while clock.get_time() < 50:
        system.update()
        np.testing.assert_allclose(
            [0.0, float(clock.get_time()), 0.0, 0.0, 0.0, 0.0, 0.0],
            np.array(system.state),
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        clock += 1
        assert nodes[2].last_received is None

    # t = 50 -- ACT (start compute)
    system.act([1])

    # t: [50, 60) -- COMPUTE
    while clock.get_time() < 60:
        system.update()
        np.testing.assert_allclose(
            [1.0, float(clock.get_time() - 50), 0.0, 0.0, 0.0, 0.0, 0.0],
            np.array(system.state),
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        clock += 1
        assert nodes[2].last_received is None

    # t = 60 -- OUTPUT RECEIVED
    system.update()
    np.testing.assert_allclose(
        [0.0, float(clock.get_time() - 50), 1.0, 60.0, 60.0, 60.0, 0.0],
        np.array(system.state),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


# TODO: Receive callback in MISONode
# TODO: Add state normalizer to Node function
# TODO: More tests for system (lexicographical sort)
# TODO: More tests for system (state)
