from computation_sim.nodes import ComputeNode
from computation_sim.basic_types import Time, NodeId, Message, Header
import pytest
from unittest.mock import Mock, MagicMock


@pytest.fixture
def setup_empty():
    clock_mock = Mock()
    clock_mock.time = 1000
    sampler_mock = Mock()
    sampler_mock.sample.return_value = 5
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
    setup_empty[0].set_output_pass(recv_pass_mock)
    return setup_empty


@pytest.fixture
def setup_none_with_outputs(setup_empty):
    node, _, _, recv_pass_mock, recv_fail_mock = setup_empty
    setup_empty[0].set_output_fail(recv_fail_mock)
    setup_empty[0].set_output_pass(recv_pass_mock)
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


def test_single_input_single_trigger(setup_empty_with_outputs):
    node, clock_mock, sampler_mock, recv_pass_mock, recv_fail_mock = (
        setup_empty_with_outputs
    )

    msg = Message(Header())
    msg.header.t_measure_average = 500
    msg.header.t_measure_youngest = 600
    msg.header.t_measure_oldest = 350
    msg.header.num_measurements = 5

    node.receive(msg)

    # Check initial state
    assert node.state[0] == pytest.approx(0.0, 1.0e-6)
    assert node.state[1] == pytest.approx(0.0, 1.0e-6)
    assert not node.is_busy
    recv_pass_mock.receive.assert_not_called()
    recv_fail_mock.receive.assert_not_called()

    # Time step
    clock_mock.time += 10
    node.update()

    # Nothing was triggered; Check that nothing changed on the state
    assert node.state[0] == pytest.approx(0.0, 1.0e-6)
    assert node.state[1] == pytest.approx(10.0, 1.0e-6)
    assert not node.is_busy
    recv_pass_mock.receive.assert_not_called()
    recv_fail_mock.receive.assert_not_called()

    # Trigger a compute
    node.trigger()

    # Increment time and check state
    for i in range(5):
        assert node.state[0] == pytest.approx(1.0, 1.0e-6)
        assert node.state[1] == pytest.approx(float(i), 1.0e-6)
        assert node.is_busy
        recv_pass_mock.receive.assert_not_called()
        recv_fail_mock.receive.assert_not_called()

        clock_mock.time += 1
        node.update()

    assert node.state[0] == pytest.approx(0.0, 1.0e-6)
    assert node.state[1] == pytest.approx(float(5), 1.0e-6)
    assert not node.is_busy
    recv_pass_mock.receive.assert_called()
    recv_fail_mock.receive.assert_not_called()


def test_no_duplicate_trigger(setup_empty_with_outputs):
    node, clock_mock, sampler_mock, recv_pass_mock, recv_fail_mock = (
        setup_empty_with_outputs
    )

    msg = Message(Header())
    msg.header.t_measure_average = 500
    msg.header.t_measure_youngest = 600
    msg.header.t_measure_oldest = 350
    msg.header.num_measurements = 5

    node.receive(msg)

    # Trigger a compute
    node.trigger()

    # Increment time and check state
    for _ in range(4):
        assert node.is_busy
        clock_mock.time += 1
        node.update()

    # Trigger while busy. Trigger should be ignored (no additional 5 sec exec. time)
    node.trigger()

    # Increment to termination time
    clock_mock.time += 1
    node.update()

    assert not node.is_busy
