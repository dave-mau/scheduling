from typing import List, Tuple
from unittest.mock import Mock

import numpy as np
import pytest
from computation_sim.basic_types import BadActionError, BadNodeGraphError
from computation_sim.system import System


def test_update_order_topologically_sorted():
    call_order = []
    nodes = [Mock() for _ in range(3)]
    nodes[0].outputs = [nodes[1]]
    nodes[1].outputs = [nodes[2]]
    nodes[2].outputs = []
    nodes[0].update.side_effect = lambda: call_order.append(0)
    nodes[1].update.side_effect = lambda: call_order.append(1)
    nodes[2].update.side_effect = lambda: call_order.append(2)

    system = System()
    # Add nodes in non-topological order
    system.add_node(nodes[1])
    system.add_node(nodes[2])
    system.add_node(nodes[0])
    system.update()

    # Make sure that the update methods were calld in topological order
    assert call_order == [0, 1, 2]


def test_update_forest_fails():
    nodes = [Mock() for _ in range(3)]
    nodes[0].outputs = [nodes[1]]
    nodes[1].outputs = []
    nodes[2].outputs = []

    system = System()
    # Add nodes in non-topological order
    system.add_node(nodes[1])
    system.add_node(nodes[2])
    system.add_node(nodes[0])

    with pytest.raises(BadNodeGraphError):
        system.update()


def test_update_cyclic_fails():
    nodes = [Mock() for _ in range(3)]
    nodes[0].outputs = [nodes[1]]
    nodes[1].outputs = [nodes[2]]
    nodes[2].outputs = [nodes[0]]

    system = System()
    # Add nodes in non-topological order
    system.add_node(nodes[1])
    system.add_node(nodes[2])
    system.add_node(nodes[0])

    with pytest.raises(BadNodeGraphError):
        system.update()


@pytest.fixture
def setup_actions():
    actions = [Mock(), Mock()]
    system = System()
    for action in actions:
        system.add_action(action)
    return system, actions


def test_act_raises(setup_actions):
    system, _ = setup_actions

    with pytest.raises(BadActionError):
        system.act([1, 2, 3])

    with pytest.raises(BadActionError):
        system.act([1])

    with pytest.raises(BadActionError):
        system.act([])


def test_act_all_numpy(setup_actions):
    system, actions = setup_actions

    system.act(np.array([1, 1]))
    actions[0].act.assert_called_once()
    actions[1].act.assert_called_once()


def test_act_some_numpy(setup_actions):
    system, actions = setup_actions

    system.act(np.array([0, 1]))
    actions[0].act.assert_not_called()
    actions[1].act.assert_called_once()


def test_act_none():
    system = System()
    system.act([])


def test_num_actions(setup_actions):
    system, actions = setup_actions
    assert system.num_action == len(actions)


def test_num_nodes():
    system = System()
    node = Mock()
    node.outputs = []
    system.add_node(node)
    node = Mock()
    node.outputs = []
    system.add_node(node)

    assert system.num_nodes == 2