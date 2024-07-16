from unittest.mock import Mock

import pytest
from computation_sim.basic_types import Header, Message
from computation_sim.nodes import SinkNode, ConstantNormalizer


def test_receive_some():
    node = SinkNode(Mock(), count_normalizer=ConstantNormalizer(10.0))
    node.receive(Message(Header(), data="foo"))
    node.receive(Message(Header(), data="bar"))

    assert len(node.state) == 1
    assert node.state[0] == pytest.approx(0.2, 1.0e-6)


def test_receive_none():
    node = SinkNode(Mock())
    assert len(node.state) == 1
    assert node.state[0] == pytest.approx(0.0, 1.0e-6)


def test_update_clears_received():
    node = SinkNode(Mock())
    node.receive(Message(Header(), data="foo"))
    node.receive(Message(Header(), data="bar"))
    node.update()
    assert len(node.state) == 1
    assert node.state[0] == pytest.approx(0.0, 1.0e-6)


def test_reset_clears_received():
    node = SinkNode(Mock())
    node.receive(Message(Header(), data="foo"))
    node.receive(Message(Header(), data="bar"))
    node.reset()
    assert len(node.state) == 1
    assert node.state[0] == pytest.approx(0.0, 1.0e-6)
