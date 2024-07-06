from computation_sim.nodes import SinkNode
from computation_sim.basic_types import Message, Header
import pytest


def test_receive_some():
    node = SinkNode()
    node.receive(Message(Header(), data="foo"))
    node.receive(Message(Header(), data="bar"))

    assert len(node.state) == 1
    assert node.state[0] == pytest.approx(2.0, 1.0e-6)


def test_receive_none():
    node = SinkNode()
    assert len(node.state) == 1
    assert node.state[0] == pytest.approx(0.0, 1.0e-6)


def test_update_clears_received():
    node = SinkNode()
    node.receive(Message(Header(), data="foo"))
    node.receive(Message(Header(), data="bar"))
    node.update(111)
    assert len(node.state) == 1
    assert node.state[0] == pytest.approx(0.0, 1.0e-6)


def test_reset_clears_received():
    node = SinkNode()
    node.receive(Message(Header(), data="foo"))
    node.receive(Message(Header(), data="bar"))
    node.reset()
    assert len(node.state) == 1
    assert node.state[0] == pytest.approx(0.0, 1.0e-6)
