from unittest.mock import Mock

import pytest
from environment import (Buffer, MessageLossCounter, ProcessingNode,
                         ProcessingState)


@pytest.fixture
def setup():
    inputs = [Buffer("In0"), Buffer("In1")]
    inputs[0].write(1)
    inputs[1].write(2)
    output = Buffer("Out")
    counter = MessageLossCounter()
    mock_compute_task = Mock()
    node = ProcessingNode(mock_compute_task, id="NODE", output=output, message_loss_counter=counter)
    node.add_input(inputs[0])
    node.add_input(inputs[1])
    return (inputs, output, counter, mock_compute_task, node)


def test_initial_state_startup(setup):
    _, _, _, _, node = setup
    assert node.state == ProcessingState.STARTUP


def test_update_startup(setup):
    _, _, _, mock_compute_task, node = setup
    node.update()
    node.update()

    mock_compute_task.assert_not_called()


def test_read_inputs(setup):
    _, _, _, _, node = setup

    result = node.read_inputs()
    assert len(result) == 2
    assert result[0] == 1
    assert result[1] == 2


def test_regular_cycle(setup):
    _, output, counter, mock_compute_task, node = setup
    mock_compute_task.compute_info.num_rejected_inputs = 2
    mock_compute_task.result = 1

    # Act: Start a compute cycle
    node.trigger_compute()
    assert counter.get_count("NODE") == 2

    # Assert: Processing cycle gets the correct input args
    mock_compute_task.run.assert_called_once_with([1, 2])
    # Assert: Node is in busy state
    assert node.state == ProcessingState.BUSY

    # Act: Update while cycle is not finished
    mock_compute_task.is_finished = False
    node.update()

    # Assert: Still busy, output receives nothing
    assert node.state == ProcessingState.BUSY
    assert output.read() is None

    # Act: Update when cycle is finished
    mock_compute_task.is_finished = True
    mock_compute_task.result = 9
    node.update()

    # Assert: State is idle again
    assert node.state == ProcessingState.IDLE
    # Assert: Output got written to
    assert output.read() == 9


def test_consecutive_trigger(setup):
    _, _, counter, mock_compute_task, node = setup
    mock_compute_task.compute_info.num_rejected_inputs = 1
    mock_compute_task.result = 2

    # Act: Start a compute cycle
    node.trigger_compute()
    assert counter.get_count("NODE") == 1

    # Assert: Processing cycle gets the correct input args
    mock_compute_task.run.assert_called_once_with([1, 2])
    # Assert: Node is in busy state
    assert node.state == ProcessingState.BUSY

    # Assert: Calling trigger compute twice will have no effect
    mock_compute_task.reset_mock()
    node.trigger_compute()
    assert counter.get_count("NODE") == 1
    mock_compute_task.run.assert_not_called()


def test_reset():
    mock_task = Mock()
    mock_out = Mock()
    node = ProcessingNode(mock_task, output=mock_out)
    node._state = ProcessingState.BUSY

    node.reset()
    mock_task.reset.assert_called_once()
    mock_out.reset.assert_called_once()
    assert node.state == ProcessingState.STARTUP
