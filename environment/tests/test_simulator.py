import pytest
from unittest.mock import Mock
from environment import Simulator, Clock
from itertools import chain


@pytest.fixture
def setup():
    clock = Clock(0)
    sim = Simulator(clock, 5)
    nodes = [[Mock(), Mock()], [Mock(), Mock()]]
    for order, nodes_order in enumerate(nodes):
        for node in nodes_order:
            sim.add_node(node, order)
    return clock, nodes, sim


def test_init(setup):
    clock, nodes, sim = setup
    for node in chain(*nodes):
        node.update.assert_not_called()
    sim.init()
    for node in chain(*nodes):
        node.update.assert_called_once()


def test_advance(setup):
    clock, nodes, sim = setup
    sim.init()
    sim.advance()

    assert clock.get_time_ms() == 5
    for node in chain(*nodes):
        assert node.update.call_count == 2


def test_run_until(setup):
    clock, nodes, sim = setup
    sim.init()
    sim.run_until(100)

    assert clock.get_time_ms() == 100
    for node in chain(*nodes):
        assert node.update.call_count == 21


def test_reset(setup):
    clock, nodes, sim = setup
    clock.advance(10)

    sim.reset()
    assert sim.time == 0
    for node in chain(*nodes):
        node.reset.assert_called_once()
