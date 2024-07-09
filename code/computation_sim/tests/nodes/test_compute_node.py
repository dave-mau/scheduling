from computation_sim.nodes import ComputeNode
from computation_sim.basic_types import Time, NodeId
import pytest
from unittest.mock import Mock, MagicMock


@pytest.fixture
def setup_empty():
    clock_mock = Mock()
    clock_mock.time = 1000
    sampler_mock = Mock()
    recv_pass_mock = Mock()
    recv_fail_mock = Mock()
    node = ComputeNode(clock_mock, sampler_mock)
    return node, clock_mock, sampler_mock, recv_pass_mock, recv_fail_mock


@pytest.fixture
def setup_none(setup_empty):
    setup_empty[0].receive(None)
    setup_empty[0].receive(None)
    return setup_empty


@pytest.fixture
def setup_empty_with_outputs(setup_empty):
    node, _, _, recv_pass_mock, recv_fail_mock = setup_empty
    setup_empty[0].set_output_fail(recv_fail_mock)
    setup_empty[1].set_output_pass(recv_pass_mock)
    return setup_empty


@pytest.fixture
def setup_none_with_outputs(setup_empty):
    node, _, _, recv_pass_mock, recv_fail_mock = setup_empty
    setup_empty[0].set_output_fail(recv_fail_mock)
    setup_empty[1].set_output_pass(recv_pass_mock)
    return setup_empty


@pytest.mark.parametrize(
    "fixture",
    [
        "setup_empty",
        "setup_none",
        "setup_empty_with_outputs",
        "setup_none_with_outputs",
    ],
)
def test_empty_inputs(request, fixture):
    fixture_value = request.getfixturevalue(fixture)
    node, clock_mock, sampler_mock, recv_pass_mock, recv_fail_mock = fixture_value

    # Check initial state
    assert node.state[0] == pytest.approx(0.0, 1.0e-6)
    assert node.state[1] == pytest.approx(0.0, 1.0e-6)
    assert not node.is_busy

    # Trigger: Check state again
    node.trigger()
    assert node.state[0] == pytest.approx(0.0, 1.0e-6)
    assert node.state[1] == pytest.approx(0.0, 1.0e-6)
    assert not node.is_busy

    # Advance time, update, check state again
    clock_mock.time += 1
    node.update()
    assert node.state[0] == pytest.approx(0.0, 1.0e-6)
    assert node.state[1] == pytest.approx(1.0, 1.0e-6)
    assert not node.is_busy

    # Assert nothing was sent to any of the receivers
    recv_fail_mock.receive.assert_not_called()
    recv_pass_mock.receive.assert_not_called()
