from unittest.mock import MagicMock, Mock

import pytest
from environment import Buffer, Clock, InputNode


@pytest.fixture
def setup():
    clock = Clock(10)
    sampler = Mock()
    buffer = Buffer("Output")
    attrs = {"sample.return_value": 0.0}
    sampler.configure_mock(**attrs)
    return (
        clock,
        buffer,
        sampler,
        InputNode(clock.as_readonly(), 100, 5, sampler, output=buffer),
    )


def test_nominal_next_trigger(setup):
    clock, buffer, sampler, node = setup
    assert node.nominal_next_trigger_time() == 105


def test_nominal_last_trigger(setup):
    clock, buffer, sampler, node = setup
    assert node.nominal_last_trigger_time() == 5


def test_next_trigger_time(setup):
    clock, buffer, sampler, node = setup

    attrs = {"sample.return_value": 3.1}
    sampler.configure_mock(**attrs)
    assert node.next_trigger_time() == 108


def test_update_follows_triggers_no_offset(setup):
    clock, buffer, sampler, node = setup
    attrs = {"sample.return_value": 0.0}
    sampler.configure_mock(**attrs)

    assert node.next_trigger_time() == 105
    while clock.get_time_ms() < 105:
        node.update()
        assert buffer.read() == None
        clock.advance(1)

    node.update()
    assert buffer.read().min_time == 105
    assert buffer.read().avg_time == 105

    while clock.get_time_ms() < 205:
        node.update()
        assert buffer.read().min_time == 105
        assert buffer.read().avg_time == 105
        clock.advance(1)

    node.update()
    assert buffer.read().min_time == 205
    assert buffer.read().avg_time == 205


def test_update_follows_triggers_offset(setup):
    clock, buffer, sampler, node = setup

    while clock.get_time_ms() < 105:
        node.update()
        assert buffer.read() == None
        clock.advance(1)

    # Update will sample next trigger
    attrs = {"sample.return_value": -2.1}
    sampler.configure_mock(**attrs)
    node.update()
    assert buffer.read().min_time == 105
    assert buffer.read().avg_time == 105

    while clock.get_time_ms() < 203:
        node.update()
        assert buffer.read().min_time == 105
        assert buffer.read().avg_time == 105
        clock.advance(1)

    # Update will sample next trigger
    attrs = {"sample.return_value": 1.1}
    sampler.configure_mock(**attrs)
    node.update()
    assert buffer.read().min_time == 203
    assert buffer.read().avg_time == 203

    while clock.get_time_ms() < 306:
        node.update()
        assert buffer.read().min_time == 203
        assert buffer.read().avg_time == 203
        clock.advance(1)


def test_reset():
    clock = Clock(0)
    mock_buffer = Mock()
    sampler = Mock()
    sampler.sample = MagicMock(return_value=10)
    node = InputNode(clock.as_readonly(), 100, 0, sampler, id="A", output=mock_buffer)

    node.reset()
    mock_buffer.reset.assert_called_once()
