from collections import namedtuple
from typing import Tuple
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
from environment import (BadSystemArchitectureError,
                         ConnectionNextNodeMissingError, DuplicateNodeError,
                         Message, StateVector, System)


def make_mock_node(id: str):
    node = Mock()
    node.id = id
    node.output = Mock()
    node.output.id = f"{id}_OUT"
    return node


@pytest.fixture
def setup_empty_system() -> Tuple[Mock, System]:
    mock_time_provider = Mock()
    mock_time_provider.time = 0
    system = System(mock_time_provider)
    return mock_time_provider, system


def test_empty_compile(setup_empty_system):
    tp, system = setup_empty_system
    system.compile()
    assert len(system.input_nodes) == 0
    assert len(system.state_nodes) == 0
    assert len(system.action_nodes) == 0


def test_compile_two_element_system(setup_empty_system):
    tp, system = setup_empty_system

    output_node = make_mock_node("OUTPUT")
    input_node = make_mock_node("INPUT")

    system.add_processing_node(output_node, None)
    system.add_input_node(input_node, "OUTPUT", trigger_next_compute=True)

    output_node.add_input.assert_called_once_with(input_node.output)
    input_node.output.register_exit_write_cb.assert_called_once()

    system.compile()
    assert len(system.input_nodes) == 1
    assert system.input_nodes[0] == input_node
    assert len(system.state_nodes) == 1
    assert system.state_nodes[0] == output_node
    assert len(system.action_nodes) == 1
    assert system.action_nodes[0] == output_node


def test_compile_three_element_system(setup_empty_system):
    tp, system = setup_empty_system

    nodes = []
    for i in range(3):
        nodes.append(make_mock_node(f"NODE_{i}"))

    system.add_processing_node(nodes[2], None)
    system.add_processing_node(nodes[1], "NODE_2")
    system.add_input_node(nodes[0], "NODE_1", trigger_next_compute=False)
    nodes[1].add_input.assert_called_once_with(nodes[0].output)
    nodes[2].add_input.assert_called_once_with(nodes[1].output)
    nodes[0].output.register_exit_write_cb.assert_not_called()

    system.compile()

    assert len(system.input_nodes) == 1
    assert system.input_nodes[0] == nodes[0]
    assert len(system.state_nodes) == 2
    assert system.state_nodes[0] == nodes[1]
    assert system.state_nodes[1] == nodes[2]
    assert len(system.action_nodes) == 2
    assert system.action_nodes[0] == nodes[1]
    assert system.action_nodes[1] == nodes[2]


def test_compile_none_action(setup_empty_system):
    tp, system = setup_empty_system

    nodes = []
    for i in range(3):
        nodes.append(make_mock_node(f"NODE_{i}"))

    system.add_processing_node(nodes[2], None, is_action_node=True, is_state_node=True)
    system.add_processing_node(nodes[1], "NODE_2", is_action_node=False, is_state_node=True)
    system.add_input_node(nodes[0], "NODE_1", trigger_next_compute=False)
    system.compile()

    assert len(system.state_nodes) == 2
    assert system.state_nodes[0] == nodes[1]
    assert system.state_nodes[1] == nodes[2]
    assert len(system.action_nodes) == 1
    assert system.action_nodes[0] == nodes[2]


def test_compile_none_state(setup_empty_system):
    tp, system = setup_empty_system

    nodes = []
    for i in range(3):
        nodes.append(make_mock_node(f"NODE_{i}"))

    system.add_processing_node(nodes[2], None, is_action_node=True, is_state_node=True)
    system.add_processing_node(nodes[1], "NODE_2", is_action_node=True, is_state_node=False)
    system.add_input_node(nodes[0], "NODE_1", trigger_next_compute=False)
    system.compile()

    assert len(system.state_nodes) == 1
    assert system.state_nodes[0] == nodes[2]
    assert len(system.action_nodes) == 2
    assert system.action_nodes[0] == nodes[1]
    assert system.action_nodes[1] == nodes[2]


def test_add_processing_node_raises_duplicate(setup_empty_system):
    _, system = setup_empty_system
    output_node = make_mock_node("OUTPUT")
    system.add_processing_node(output_node, None)
    with pytest.raises(DuplicateNodeError):
        system.add_processing_node(output_node, "OUTPUT")
    with pytest.raises(DuplicateNodeError):
        system.add_input_node(output_node, "OUTPUT")


def test_add_processing_node_raises_missing_connection(setup_empty_system):
    _, system = setup_empty_system
    output_node = make_mock_node("OUTPUT")
    other_node = make_mock_node("OTHER")
    with pytest.raises(ConnectionNextNodeMissingError):
        system.add_processing_node(other_node, "ABC")
    with pytest.raises(ConnectionNextNodeMissingError):
        system.add_input_node(other_node, "ABC")


@pytest.fixture
def setup_simple_tree_system():
    tp = Mock()
    system = System(tp)
    nodes = {}

    nodes["LEVEL2"] = make_mock_node("LEVEL2")
    system.add_processing_node(nodes["LEVEL2"], None)
    for i in range(2):
        nodes[f"LEVEL1_{i}"] = make_mock_node(f"LEVEL1_{i}")
        system.add_processing_node(nodes[f"LEVEL1_{i}"], "LEVEL2")
        for j in range(2):
            nodes[f"INPUT_{i}{j}"] = make_mock_node(f"INPUT_{i}{j}")
            system.add_input_node(nodes[f"INPUT_{i}{j}"], f"LEVEL1_{i}")
    system.compile()

    return tp, system, nodes


def test_get_state_without_output(setup_simple_tree_system):
    tp, system, nodes = setup_simple_tree_system
    for idx, val in enumerate(system.state_nodes):
        val.compute_task.t_start = idx
        val.compute_task.is_running = True
        val.output.has_element = False

    # Age is just negative compute task time
    tp.time = 0
    result = system.get_state()

    assert pytest.approx(result.compute_start_ages[0], 1.0e-6) == -(nodes["LEVEL1_0"].compute_task.t_start / 100.0)
    assert pytest.approx(result.compute_start_ages[1], 1.0e-6) == -(nodes["LEVEL1_1"].compute_task.t_start / 100.0)
    assert pytest.approx(result.compute_start_ages[2], 1.0e-6) == -(nodes["LEVEL2"].compute_task.t_start / 100.0)
    for i in range(3):
        assert result.compute_running[i]
        assert not result.buf_out_has_value[i]


def test_get_state_with_output(setup_simple_tree_system):
    tp, system, nodes = setup_simple_tree_system
    for idx, val in enumerate(system.state_nodes):
        val.compute_task.t_start = idx
        val.compute_task.is_running = True
        val.output.has_element = True
        val.output.read = MagicMock(return_value=Message(10 * idx, 20 * idx, 30 * idx))

    # Age is just negative compute task time
    tp.time = 0
    result = system.get_state()

    assert pytest.approx(result.compute_start_ages[0], 1.0e-6) == -(nodes["LEVEL1_0"].compute_task.t_start / 100.0)
    assert pytest.approx(result.buf_out_min_ages[0], 1.0e-6) == -10 * (nodes["LEVEL1_0"].compute_task.t_start / 100.0)
    assert pytest.approx(result.buf_out_avg_ages[0], 1.0e-6) == -20 * (nodes["LEVEL1_0"].compute_task.t_start / 100.0)
    assert pytest.approx(result.buf_out_max_ages[0], 1.0e-6) == -30 * (nodes["LEVEL1_0"].compute_task.t_start / 100.0)

    assert pytest.approx(result.compute_start_ages[1], 1.0e-6) == -(nodes["LEVEL1_1"].compute_task.t_start / 100.0)
    assert pytest.approx(result.buf_out_min_ages[1], 1.0e-6) == -10 * (nodes["LEVEL1_1"].compute_task.t_start / 100.0)
    assert pytest.approx(result.buf_out_avg_ages[1], 1.0e-6) == -20 * (nodes["LEVEL1_1"].compute_task.t_start / 100.0)
    assert pytest.approx(result.buf_out_max_ages[1], 1.0e-6) == -30 * (nodes["LEVEL1_1"].compute_task.t_start / 100.0)

    assert pytest.approx(result.compute_start_ages[2], 1.0e-6) == -(nodes["LEVEL2"].compute_task.t_start / 100.0)
    assert pytest.approx(result.buf_out_min_ages[2], 1.0e-6) == -10 * (nodes["LEVEL2"].compute_task.t_start / 100.0)
    assert pytest.approx(result.buf_out_avg_ages[2], 1.0e-6) == -20 * (nodes["LEVEL2"].compute_task.t_start / 100.0)
    assert pytest.approx(result.buf_out_max_ages[2], 1.0e-6) == -30 * (nodes["LEVEL2"].compute_task.t_start / 100.0)

    for i in range(3):
        assert result.compute_running[i]
        assert result.buf_out_has_value[i]


def test_act_000(setup_simple_tree_system):
    tp, system, nodes = setup_simple_tree_system

    system.act(np.zeros((3,)))
    for val in nodes.values():
        val.trigger_compute.assert_not_called()


def test_act_100(setup_simple_tree_system):
    tp, system, nodes = setup_simple_tree_system
    system.act(np.array([1, 0, 0]))
    nodes["LEVEL1_0"].trigger_compute.assert_called_once()
    nodes["LEVEL1_1"].trigger_compute.assert_not_called()
    nodes["LEVEL2"].trigger_compute.assert_not_called()


def test_act_010(setup_simple_tree_system):
    tp, system, nodes = setup_simple_tree_system
    system.act(np.array([0, 1, 0]))
    nodes["LEVEL1_0"].trigger_compute.assert_not_called()
    nodes["LEVEL1_1"].trigger_compute.assert_called_once()
    nodes["LEVEL2"].trigger_compute.assert_not_called()


def test_act_001(setup_simple_tree_system):
    tp, system, nodes = setup_simple_tree_system
    system.act(np.array([0, 0, 1]))
    nodes["LEVEL1_0"].trigger_compute.assert_not_called()
    nodes["LEVEL1_1"].trigger_compute.assert_not_called()
    nodes["LEVEL2"].trigger_compute.assert_called_once()


def test_act_111(setup_simple_tree_system):
    tp, system, nodes = setup_simple_tree_system
    system.act(np.array([1, 1, 1]))
    nodes["LEVEL1_0"].trigger_compute.assert_called_once()
    nodes["LEVEL1_1"].trigger_compute.assert_called_once()
    nodes["LEVEL2"].trigger_compute.assert_called_once()


def test_num_properties(setup_simple_tree_system):
    tp, system, nodes = setup_simple_tree_system
    assert system.num_state_nodes == 3
    assert system.num_action_nodes == 3
    assert system.num_states == 18
