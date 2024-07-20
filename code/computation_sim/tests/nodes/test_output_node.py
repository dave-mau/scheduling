from unittest.mock import MagicMock, Mock

import pytest
from computation_sim.basic_types import Header, Message
from computation_sim.nodes import ConstantNormalizer, OutputNode


def test_receive_copies():
    node = OutputNode(Mock())
    assert node.last_received is None
    node.receive(Message(Header(), "foo"))
    assert node.last_received is not None
    assert node.last_received.data == "foo"


def test_receive_calls_callback():
    mock_cb = MagicMock()
    node = OutputNode(Mock(), receive_cb=mock_cb)
    node.receive(Message(Header(), "foo"))

    assert mock_cb.called
    assert mock_cb.call_args[0][0].data == "foo"


def test_state_has_output():
    clock_mock = Mock()
    clock_mock.time = 10

    node = OutputNode(
        clock_mock,
        age_normalizer=ConstantNormalizer(10.0),
        occupancy_normalizer=ConstantNormalizer(0.1),
    )
    node.receive(Message(Header(1, 2, 3), "foo"))

    state = node.state
    assert state[0] == pytest.approx(10.0, 1.0e-6)
    assert state[1] == pytest.approx(0.9, 1.0e-6)
    assert state[2] == pytest.approx(0.8, 1.0e-6)
    assert state[3] == pytest.approx(0.7, 1.0e-6)

    clock_mock.time = 11
    state = node.state
    assert state[0] == pytest.approx(10.0, 1.0e-6)
    assert state[1] == pytest.approx(1.0, 1.0e-6)
    assert state[2] == pytest.approx(0.9, 1.0e-6)
    assert state[3] == pytest.approx(0.8, 1.0e-6)


def test_state_has_no_output():
    clock_mock = Mock()
    clock_mock.time = 10

    node = OutputNode(clock_mock)

    state = node.state
    assert state[0] == pytest.approx(0.0, 1.0e-6)
    assert state[1] == pytest.approx(0.0, 1.0e-6)
    assert state[2] == pytest.approx(0.0, 1.0e-6)
    assert state[3] == pytest.approx(0.0, 1.0e-6)


def test_reset():
    clock_mock = Mock()
    clock_mock.time = 10

    node = OutputNode(clock_mock)
    node.receive(Message(Header(1, 2, 3), "foo"))

    assert node.state[0] == pytest.approx(1.0, 1.0e-6)

    node.reset()
    assert node.last_received is None
    assert node.state[0] == pytest.approx(0.0, 1.0e-6)
