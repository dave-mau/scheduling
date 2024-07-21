from typing import Tuple

import numpy as np
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
def setup() -> Tuple[System, Clock, Tuple[Node]]:
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
        age_normalizer=normalizer,
    )
    buffer_1 = RingBufferNode(
        clock.as_readonly(),
        "BUFFER_1",
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
    buffer_0.set_output(compute_node)
    buffer_0.set_overflow_output(lost_node)
    buffer_1.set_output(compute_node)
    buffer_1.set_overflow_output(lost_node)
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


def test_no_action(setup):
    system, clock, _ = setup

    # Run clock for first 99 time units
    while clock.get_time() < 100:
        # On first update, sensor immediately writes output to buffers.
        system.update()
        age = clock.get_time() / 100.0
        expected_state = np.array(
            [
                1.0,  #  0: Buffer 0: Occupancy
                age,  #  1: Buffer 0: Oldest Age
                age,  #  2: Buffer 0: Youngest Age
                age,  #  3: Buffer 0: Average Age
                1.0,  #  4: Buffer 1: Occupancy
                age,  #  5: Buffer 1: Oldest Age
                age,  #  6: Buffer 1: Youngest Age
                age,  #  7: Buffer 1: Average Age
                0.0,  #  8: Compute Node: Busy
                age,  #  9: Compute Node: Age last started
                0.0,  # 10: Lost Messages: Count
                0.0,  # 11: Output: Occupancy
                0.0,  # 12: Output: Oldest Age
                0.0,  # 13: Output: Youngest Age
                0.0,  # 14: Output: Average Age
            ]
        )
        state = np.array(system.state)
        np.testing.assert_allclose(expected_state, state)
        clock += 10

    # Sensors trigger at t=100.
    while clock.get_time() < 200:
        # Upon update, buffers will receive new messages.
        # The overflow (old) messages should be written to the lost-node.
        system.update()
        age = (clock.get_time() - 100) / 100.0
        expected_state = np.array(
            [
                1.0,  #  0: Buffer 0: Occupancy
                age,  #  1: Buffer 0: Oldest Age
                age,  #  2: Buffer 0: Youngest Age
                age,  #  3: Buffer 0: Average Age
                1.0,  #  4: Buffer 1: Occupancy
                age,  #  5: Buffer 1: Oldest Age
                age,  #  6: Buffer 1: Youngest Age
                age,  #  7: Buffer 1: Average Age
                0.0,  #  8: Compute Node: Busy
                age + 1.0,  #  9: Compute Node: Age last started
                2.0,  # 10: Lost Messages: Count
                0.0,  # 11: Output: Occupancy
                0.0,  # 12: Output: Oldest Age
                0.0,  # 13: Output: Youngest Age
                0.0,  # 14: Output: Average Age
            ]
        )
        state = np.array(system.state)
        np.testing.assert_allclose(expected_state, state)
        clock += 10


def test_nominal(setup):
    system, clock, _ = setup

    # t in [0, 50): Nothing
    while clock.get_time() < 50:
        system.update()
        age = clock.get_time() / 100.0
        expected_state = np.array(
            [
                1.0,  #  0: Buffer 0: Occupancy
                age,  #  1: Buffer 0: Oldest Age
                age,  #  2: Buffer 0: Youngest Age
                age,  #  3: Buffer 0: Average Age
                1.0,  #  4: Buffer 1: Occupancy
                age,  #  5: Buffer 1: Oldest Age
                age,  #  6: Buffer 1: Youngest Age
                age,  #  7: Buffer 1: Average Age
                0.0,  #  8: Compute Node: Busy
                age,  #  9: Compute Node: Age last started
                0.0,  # 10: Lost Messages: Count
                0.0,  # 11: Output: Occupancy
                0.0,  # 12: Output: Oldest Age
                0.0,  # 13: Output: Youngest Age
                0.0,  # 14: Output: Average Age
            ]
        )
        state = np.array(system.state)
        np.testing.assert_allclose(expected_state, state)
        clock += 10

    # t == 50: Start compute task
    system.act([1])

    # t in [50, 60): Compute task running
    while clock.get_time() < 60:
        system.update()
        age = clock.get_time() / 100.0
        expected_state = np.array(
            [
                0.0,  #  0: Buffer 0: Occupancy
                0.0,  #  1: Buffer 0: Oldest Age
                0.0,  #  2: Buffer 0: Youngest Age
                0.0,  #  3: Buffer 0: Average Age
                0.0,  #  4: Buffer 1: Occupancy
                0.0,  #  5: Buffer 1: Oldest Age
                0.0,  #  6: Buffer 1: Youngest Age
                0.0,  #  7: Buffer 1: Average Age
                1.0,  #  8: Compute Node: Busy
                0.01 * (clock.get_time() - 50),  #  9: Compute Node: Age last started
                0.0,  # 10: Lost Messages: Count
                0.0,  # 11: Output: Occupancy
                0.0,  # 12: Output: Oldest Age
                0.0,  # 13: Output: Youngest Age
                0.0,  # 14: Output: Average Age
            ]
        )
        state = np.array(system.state)
        np.testing.assert_allclose(expected_state, state)
        clock += 1

    # t in [60, 100): Idle
    while clock.get_time() < 100:
        system.update()
        age = clock.get_time() / 100.0
        expected_state = np.array(
            [
                0.0,  #  0: Buffer 0: Occupancy
                0.0,  #  1: Buffer 0: Oldest Age
                0.0,  #  2: Buffer 0: Youngest Age
                0.0,  #  3: Buffer 0: Average Age
                0.0,  #  4: Buffer 1: Occupancy
                0.0,  #  5: Buffer 1: Oldest Age
                0.0,  #  6: Buffer 1: Youngest Age
                0.0,  #  7: Buffer 1: Average Age
                0.0,  #  8: Compute Node: Busy
                0.01 * (clock.get_time() - 50),  #  9: Compute Node: Age last started
                0.0,  # 10: Lost Messages: Count
                1.0,  # 11: Output: Occupancy
                0.01 * clock.get_time(),  # 12: Output: Oldest Age
                0.01 * clock.get_time(),  # 13: Output: Youngest Age
                0.01 * clock.get_time(),  # 14: Output: Average Age
            ]
        )
        state = np.array(system.state)
        np.testing.assert_allclose(expected_state, state)
        clock += 10
