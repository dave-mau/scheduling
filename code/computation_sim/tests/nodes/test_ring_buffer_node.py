from unittest.mock import MagicMock, Mock

import pytest
from computation_sim.basic_types import Header, Message
from computation_sim.nodes import ConstantNormalizer, RingBufferNode


@pytest.fixture
def setup_no_callback():
    clock_mock = Mock()
    buffer = RingBufferNode(clock_mock, max_num_elements=2)
    return buffer, clock_mock


def test_receive_no_overflow(setup_no_callback):
    buffer, clock_mock = setup_no_callback

    assert buffer.num_entries == 0

    buffer.receive(Message(Header(1, 2, 3)))
    assert buffer.num_entries == 1

    buffer.receive(Message(Header(4, 5, 6)))
    assert buffer.num_entries == 2

    result = buffer.pop()
    assert result.header.t_measure_oldest == 1

    result = buffer.pop()
    assert result.header.t_measure_oldest == 4

    result = buffer.pop()
    assert result is None


def test_receive_overflow(setup_no_callback):
    buffer, clock_mock = setup_no_callback

    assert buffer.num_entries == 0

    buffer.receive(Message(Header(1, 2, 3)))
    assert buffer.num_entries == 1

    buffer.receive(Message(Header(4, 5, 6)))
    assert buffer.num_entries == 2

    buffer.receive(Message(Header(7, 8, 9)))
    assert buffer.num_entries == 2

    result = buffer.pop()
    assert result.header.t_measure_oldest == 4

    result = buffer.pop()
    assert result.header.t_measure_oldest == 7

    result = buffer.pop()
    assert result is None


@pytest.fixture
def setup_callback():
    clock_mock = Mock()
    cb_mock = MagicMock()
    buffer = RingBufferNode(clock_mock, max_num_elements=2, overflow_cb=cb_mock)
    return buffer, clock_mock, cb_mock


def test_receive_no_overflow(setup_callback):
    buffer, _, cb_mock = setup_callback

    buffer.receive(Message(Header(1, 2, 3)))
    buffer.receive(Message(Header(4, 5, 6)))
    assert not cb_mock.called


def test_receive_overflow(setup_callback):
    buffer, _, cb_mock = setup_callback

    buffer.receive(Message(Header(1, 2, 3)))
    buffer.receive(Message(Header(4, 5, 6)))
    buffer.receive(Message(Header(5, 6, 7)))
    assert cb_mock.call_count == 1
    assert cb_mock.call_args[0][0].header.t_measure_oldest == 1
    buffer.receive(Message(Header(8, 9, 10)))
    assert cb_mock.call_count == 2
    assert cb_mock.call_args[0][0].header.t_measure_oldest == 4


def test_trigger_empty(setup_no_callback):
    buffer, _ = setup_no_callback
    output_mock = Mock()
    buffer.add_output(output_mock)

    buffer.trigger()

    assert output_mock.receive.call_count == 0


def test_trigger_full(setup_no_callback):
    buffer, _ = setup_no_callback
    output_mock = Mock()
    buffer.add_output(output_mock)

    buffer.receive(Message(Header(1, 1, 1)))
    buffer.receive(Message(Header(2, 2, 2)))
    buffer.receive(Message(Header(3, 3, 3)))

    buffer.trigger()

    assert output_mock.receive.call_count == 1
    assert output_mock.receive.call_args[0][0].header.t_measure_oldest == 2


def test_pop(setup_no_callback):
    buffer, _ = setup_no_callback
    buffer.receive(Message(Header(1, 1, 1)))
    buffer.receive(Message(Header(2, 2, 2)))

    result = buffer.pop()
    assert result is not None
    assert result.header.t_measure_average == 1

    result = buffer.pop()
    assert result is not None
    assert result.header.t_measure_average == 2

    result = buffer.pop()
    assert result is None


def test_reset(setup_no_callback):
    buffer, _ = setup_no_callback

    buffer.receive(Message(Header(1, 1, 1)))
    assert buffer.num_entries == 1
    buffer.reset()
    assert buffer.num_entries == 0


def test_state_full_default_normalizer(setup_no_callback):
    buffer, clock_mock = setup_no_callback
    clock_mock.time = 6
    buffer.receive(Message(Header(1, 2, 3)))
    buffer.receive(Message(Header(4, 5, 6)))

    result = buffer.generate_state()
    assert next(result) == pytest.approx(1.0, 1.0e-6)
    assert next(result) == pytest.approx(5.0, 1.0e-6)
    assert next(result) == pytest.approx(4.0, 1.0e-6)
    assert next(result) == pytest.approx(3.0, 1.0e-6)
    assert next(result) == pytest.approx(1.0, 1.0e-6)
    assert next(result) == pytest.approx(2.0, 1.0e-6)
    assert next(result) == pytest.approx(1.0, 1.0e-6)
    assert next(result) == pytest.approx(0.0, 1.0e-6)
    with pytest.raises(StopIteration):
        next(result)


def test_state_empty_default_normalizer(setup_no_callback):
    buffer, clock_mock = setup_no_callback
    clock_mock.time = 6

    result = buffer.generate_state()
    assert next(result) == pytest.approx(0.0, 1.0e-6)
    assert next(result) == pytest.approx(0.0, 1.0e-6)
    assert next(result) == pytest.approx(0.0, 1.0e-6)
    assert next(result) == pytest.approx(0.0, 1.0e-6)
    assert next(result) == pytest.approx(0.0, 1.0e-6)
    assert next(result) == pytest.approx(0.0, 1.0e-6)
    assert next(result) == pytest.approx(0.0, 1.0e-6)
    assert next(result) == pytest.approx(0.0, 1.0e-6)
    with pytest.raises(StopIteration):
        next(result)


def test_state_semifull_default_normalizer(setup_no_callback):
    buffer, clock_mock = setup_no_callback
    clock_mock.time = 6
    buffer.receive(Message(Header(1, 2, 3)))

    result = buffer.generate_state()
    assert next(result) == pytest.approx(1.0, 1.0e-6)
    assert next(result) == pytest.approx(5.0, 1.0e-6)
    assert next(result) == pytest.approx(4.0, 1.0e-6)
    assert next(result) == pytest.approx(3.0, 1.0e-6)
    assert next(result) == pytest.approx(0.0, 1.0e-6)
    assert next(result) == pytest.approx(0.0, 1.0e-6)
    assert next(result) == pytest.approx(0.0, 1.0e-6)
    assert next(result) == pytest.approx(0.0, 1.0e-6)
    with pytest.raises(StopIteration):
        next(result)


def test_state_full_custom_normalizer():
    clock_mock = Mock()
    buffer = RingBufferNode(
        clock_mock,
        max_num_elements=2,
        age_normalizer=ConstantNormalizer(10.0),
        occupancy_normalizer=ConstantNormalizer(2.0),
    )

    clock_mock.time = 6
    buffer.receive(Message(Header(1, 2, 3)))
    buffer.receive(Message(Header(4, 5, 6)))

    result = buffer.generate_state()
    assert next(result) == pytest.approx(0.5, 1.0e-6)
    assert next(result) == pytest.approx(0.5, 1.0e-6)
    assert next(result) == pytest.approx(0.4, 1.0e-6)
    assert next(result) == pytest.approx(0.3, 1.0e-6)
    assert next(result) == pytest.approx(0.5, 1.0e-6)
    assert next(result) == pytest.approx(0.2, 1.0e-6)
    assert next(result) == pytest.approx(0.1, 1.0e-6)
    assert next(result) == pytest.approx(0.0, 1.0e-6)
    with pytest.raises(StopIteration):
        next(result)
    assert len(buffer.state) == 8
