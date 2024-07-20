from unittest.mock import Mock

import pytest
from computation_sim.basic_types import Header, Message
from computation_sim.nodes import ConstantNormalizer, SinkNode


def test_receive_some():
    clock = Mock()
    clock.time = 11
    node = SinkNode(clock, count_normalizer=ConstantNormalizer(10.0))
    node.receive(Message(Header(), data="foo"))
    node.receive(Message(Header(), data="bar"))

    assert len(node.state) == 1
    assert node.state[0] == pytest.approx(0.2, 1.0e-6)
    assert len(node.received_messages) == 2
    assert node.received_times == [11, 11]


def test_receive_none():
    node = SinkNode(Mock())
    assert len(node.state) == 1
    assert node.state[0] == pytest.approx(0.0, 1.0e-6)
    assert node.received_messages == []
    assert node.received_times == []


def test_reset_clears_received():
    node = SinkNode(Mock())
    node.receive(Message(Header(), data="foo"))
    node.receive(Message(Header(), data="bar"))
    node.reset()
    assert len(node.state) == 1
    assert node.state[0] == pytest.approx(0.0, 1.0e-6)
    assert node.received_messages == []
    assert node.received_times == []
