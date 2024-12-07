from unittest.mock import MagicMock, Mock

import pytest
from computation_sim.basic_types import Header, Message
from computation_sim.nodes import FilteringMISONode


@pytest.fixture
def setup_empty():
    clock_mock = Mock()
    clock_mock.time = 1000
    sampler_mock = Mock()
    sampler_mock.sample.return_value = 5
    recv_pass_mock = Mock()
    recv_fail_mock = Mock()
    node = FilteringMISONode(clock_mock, sampler_mock)
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
    assert len(node.state) == 4
    assert node.state[0] == pytest.approx(0.0, 1.0e-6)
    assert node.state[1] == pytest.approx(0.0, 1.0e-6)
    assert node.state[2] == pytest.approx(0.0, 1.0e-6)
    assert node.state[3] == pytest.approx(0.0, 1.0e-6)
    assert not node.is_busy

    # Trigger: Check state again
    node.trigger()
    assert len(node.state) == 4
    assert node.state[0] == pytest.approx(0.0, 1.0e-6)
    assert node.state[1] == pytest.approx(0.0, 1.0e-6)
    assert node.state[2] == pytest.approx(0.0, 1.0e-6)
    assert node.state[3] == pytest.approx(0.0, 1.0e-6)
    assert not node.is_busy

    # Advance time, update, check state again
    clock_mock.time += 1
    node.update()
    assert len(node.state) == 4
    assert node.state[0] == pytest.approx(0.0, 1.0e-6)
    assert node.state[1] == pytest.approx(1.0, 1.0e-6)
    assert node.state[2] == pytest.approx(0.0, 1.0e-6)
    assert node.state[3] == pytest.approx(0.0, 1.0e-6)
    assert not node.is_busy

    # Assert nothing was sent to any of the receivers
    recv_fail_mock.receive.assert_not_called()
    recv_pass_mock.receive.assert_not_called()


def test_single_input_single_trigger(setup_empty_with_outputs):
    node, clock_mock, sampler_mock, recv_pass_mock, recv_fail_mock = setup_empty_with_outputs

    msg = Message(Header())
    msg.header.t_measure_average = 500
    msg.header.t_measure_youngest = 600
    msg.header.t_measure_oldest = 350
    msg.header.num_measurements = 5

    node.receive(msg)

    # Check initial state
    assert len(node.state) == 4
    assert node.state[0] == pytest.approx(0.0, 1.0e-6)
    assert node.state[1] == pytest.approx(0.0, 1.0e-6)
    assert node.state[2] == pytest.approx(0.0, 1.0e-6)
    assert node.state[3] == pytest.approx(0.0, 1.0e-6)
    assert not node.is_busy
    recv_pass_mock.receive.assert_not_called()
    recv_fail_mock.receive.assert_not_called()

    # Time step
    clock_mock.time += 10
    node.update()

    # Nothing was triggered; Check that nothing changed on the state
    assert len(node.state) == 4
    assert node.state[0] == pytest.approx(0.0, 1.0e-6)
    assert node.state[1] == pytest.approx(10.0, 1.0e-6)
    assert node.state[2] == pytest.approx(0.0, 1.0e-6)
    assert node.state[3] == pytest.approx(0.0, 1.0e-6)
    assert not node.is_busy
    recv_pass_mock.receive.assert_not_called()
    recv_fail_mock.receive.assert_not_called()

    # Trigger a compute
    node.trigger()

    # Increment time and check state
    for i in range(5):
        assert len(node.state) == 4
        assert node.state[0] == pytest.approx(1.0, 1.0e-6)
        assert node.state[1] == pytest.approx(float(i), 1.0e-6)
        assert node.state[2] == pytest.approx(1.0, 1.0e-6)
        assert node.state[3] == pytest.approx(5.0, 1.0e-6)
        assert node.is_busy
        recv_pass_mock.receive.assert_not_called()
        recv_fail_mock.receive.assert_not_called()

        clock_mock.time += 1
        node.update()

    assert len(node.state) == 4
    assert node.state[0] == pytest.approx(0.0, 1.0e-6)
    assert node.state[1] == pytest.approx(float(5), 1.0e-6)
    assert node.state[2] == pytest.approx(0.0, 1.0e-6)
    assert node.state[3] == pytest.approx(0.0, 1.0e-6)
    assert not node.is_busy
    recv_pass_mock.receive.assert_called()
    recv_fail_mock.receive.assert_not_called()


def test_no_duplicate_trigger(setup_empty_with_outputs):
    node, clock_mock, sampler_mock, recv_pass_mock, recv_fail_mock = setup_empty_with_outputs

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


def test_multiple_inputs_all_accepted(setup_empty_with_outputs):
    node, clock_mock, sampler_mock, recv_pass_mock, recv_fail_mock = setup_empty_with_outputs
    msg = Message(Header())
    msg.header.t_measure_average = 500
    msg.header.t_measure_youngest = 600
    msg.header.t_measure_oldest = 350
    msg.header.num_measurements = 5
    node.receive(msg)

    msg = Message(Header())
    msg.header.t_measure_average = 600
    msg.header.t_measure_youngest = 601
    msg.header.t_measure_oldest = 360
    msg.header.num_measurements = 3
    node.receive(msg)

    node.trigger()
    clock_mock.time += 10
    node.update()

    recv_pass_mock.receive.assert_called()
    result: Message = recv_pass_mock.receive.call_args[0][0]
    assert result.header.t_measure_oldest == 350
    assert result.header.t_measure_youngest == 601
    assert result.header.num_measurements == 8
    assert result.header.t_measure_average == 538

    recv_fail_mock.receive.assert_not_called()


def test_multiple_inputs_some_rejected(setup_empty_with_outputs):
    node, clock_mock, sampler_mock, recv_pass_mock, recv_fail_mock = setup_empty_with_outputs
    msg = Message(Header())
    msg.header.t_measure_average = 500
    msg.header.t_measure_youngest = 600
    msg.header.t_measure_oldest = 410
    msg.header.num_measurements = 5
    node.receive(msg)

    msg = Message(Header())
    msg.header.t_measure_average = 490
    msg.header.t_measure_youngest = 590
    msg.header.t_measure_oldest = 400
    msg.header.num_measurements = 3
    node.receive(msg)

    msg = Message(Header())
    msg.header.t_measure_average = 490
    msg.header.t_measure_youngest = 590
    msg.header.t_measure_oldest = 390
    msg.header.num_measurements = 10
    node.receive(msg)

    node.filter_threshold = 200

    node.trigger()
    clock_mock.time += 10
    node.update()

    recv_pass_mock.receive.assert_called()
    result: Message = recv_pass_mock.receive.call_args[0][0]
    assert result.header.t_measure_oldest == 400
    assert result.header.t_measure_youngest == 600
    assert result.header.num_measurements == 8
    assert result.header.t_measure_average == 496

    recv_fail_mock.receive.assert_called()
    assert recv_fail_mock.receive.call_count == 1


def test_multiple_inputs_all_rejected(setup_empty_with_outputs):
    node, clock_mock, sampler_mock, recv_pass_mock, recv_fail_mock = setup_empty_with_outputs

    msg = Message(Header())
    msg.header.t_measure_average = 500
    msg.header.t_measure_youngest = 600
    msg.header.t_measure_oldest = 390
    msg.header.num_measurements = 5
    node.receive(msg)

    msg = Message(Header())
    msg.header.t_measure_average = 490
    msg.header.t_measure_youngest = 590
    msg.header.t_measure_oldest = 290
    msg.header.num_measurements = 3
    node.receive(msg)

    node.filter_threshold = 200

    node.trigger()
    clock_mock.time += 1
    node.update()

    recv_pass_mock.receive.assert_not_called()
    recv_fail_mock.receive.assert_called()
    assert recv_fail_mock.receive.call_count == 2

    clock_mock.time += 10
    node.update()
    recv_pass_mock.receive.assert_not_called()


def test_receive_cb_setter():
    node = FilteringMISONode(MagicMock(), MagicMock())
    trigger_cb = MagicMock()
    node.set_receive_cb(trigger_cb)
    node.receive(Message(Header()))
    trigger_cb.assert_called_once_with(node)


def test_receive_cb_init():
    trigger_cb = MagicMock()
    node = FilteringMISONode(MagicMock(), MagicMock(), receive_cb=trigger_cb)
    node.receive(Message(Header()))

    trigger_cb.assert_called_once_with(node)
