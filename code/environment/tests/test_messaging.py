from unittest.mock import MagicMock

import pytest
from environment import Buffer, Message, MessageLossCounter


@pytest.fixture
def buffer() -> Buffer:
    return Buffer("ABC")


def test_write_does_deepcopy(buffer):
    element = {"value": 0}
    buffer.write(element)

    element["value"] = 1
    assert buffer.read()["value"] == 0


def test_read_does_deepcopy(buffer):
    element_write = {"value": 0}
    buffer.write(element_write)

    element_read = buffer.read()
    buffer.write({"value": 1})
    assert element_read["value"] == 0


def test_enter_write_cb():
    mock_enter = MagicMock()
    buffer = Buffer("Input", enter_write_cb=mock_enter)
    buffer.write("A")

    mock_enter.assert_called_once_with(None)


def test_exit_write_cb():
    mock_exit = MagicMock()
    buffer = Buffer("Input", exit_write_cb=mock_exit)
    buffer.write("A")

    mock_exit.assert_called_once_with("A")


def test_message_loss():
    counter = MessageLossCounter()
    buffer = Buffer("INPUT", message_loss_counter=counter)
    buffer.write(Message(1, 2, 3))
    assert counter.get_count("INPUT") == 0

    buffer.write(Message(1, 2, 3))
    assert counter.get_count("INPUT") == 1
    buffer.write(Message(1, 2, 3))
    assert counter.get_count("INPUT") == 2

    counter.reset()
    buffer.write(Message(1, 2, 3))
    assert counter.get_count("INPUT") == 1

    buffer.reset()
    counter.reset()
    buffer.write(Message(1, 2, 3))
    assert counter.get_count("INPUT") == 0
