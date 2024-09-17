from unittest.mock import MagicMock, Mock

import pytest
from computation_sim.basic_types import BadNodeGraphError, Header, Message
from computation_sim.nodes import ConstantNormalizer, RingBufferNode


@pytest.fixture
def setup_no_outputs():
    clock_mock = Mock()
    buffer = RingBufferNode(clock_mock, max_num_elements=2)
    return buffer, clock_mock


def test_receive_no_overflow_0(setup_no_outputs):
    buffer, _ = setup_no_outputs

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


def test_receive_overflow_0(setup_no_outputs):
    buffer, _ = setup_no_outputs

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


def test_trigger_raises(setup_no_outputs):
    buffer, _ = setup_no_outputs
    with pytest.raises(BadNodeGraphError):
        buffer.trigger()


@pytest.fixture
def setup_outputs():
    clock_mock = MagicMock()
    output_mock = MagicMock()
    overflow_mock = MagicMock()
    buffer = RingBufferNode(clock_mock, max_num_elements=2)
    buffer.set_output(output_mock)
    buffer.set_overflow_output(overflow_mock)
    return buffer, clock_mock, output_mock, overflow_mock


def test_receive_no_overflow_1(setup_outputs):
    buffer, _, _, overflow_mock = setup_outputs

    buffer.receive(Message(Header(1, 2, 3)))
    buffer.receive(Message(Header(4, 5, 6)))
    assert not overflow_mock.receive.called


def test_receive_overflow_1(setup_outputs):
    buffer, _, _, overflow_mock = setup_outputs

    buffer.receive(Message(Header(1, 2, 3)))
    buffer.receive(Message(Header(4, 5, 6)))
    buffer.receive(Message(Header(5, 6, 7)))
    assert overflow_mock.receive.call_count == 1
    assert overflow_mock.receive.call_args[0][0].header.t_measure_oldest == 1
    buffer.receive(Message(Header(8, 9, 10)))
    assert overflow_mock.receive.call_count == 2
    assert overflow_mock.receive.call_args[0][0].header.t_measure_oldest == 4


def test_trigger_empty(setup_outputs):
    buffer, _, output_mock, _ = setup_outputs

    buffer.trigger()
    assert output_mock.receive.call_count == 0


def test_trigger_full(setup_outputs):
    buffer, _, output_mock, _ = setup_outputs
    buffer.receive(Message(Header(1, 1, 1)))
    buffer.receive(Message(Header(2, 2, 2)))
    buffer.receive(Message(Header(3, 3, 3)))
    buffer.trigger()

    assert output_mock.receive.call_count == 1
    assert output_mock.receive.call_args[0][0].header.t_measure_oldest == 2


def test_pop(setup_no_outputs):
    buffer, _ = setup_no_outputs
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


def test_reset(setup_no_outputs):
    buffer, _ = setup_no_outputs

    buffer.receive(Message(Header(1, 1, 1)))
    assert buffer.num_entries == 1
    buffer.reset()
    assert buffer.num_entries == 0


def test_state_full_default_normalizer(setup_no_outputs):
    buffer, clock_mock = setup_no_outputs
    clock_mock.time = 6
    buffer.receive(Message(Header(1, 2, 3, 7)))
    buffer.receive(Message(Header(4, 5, 6, 8)))

    result = buffer.generate_state()
    assert next(result) == pytest.approx(1.0, 1.0e-6)
    assert next(result) == pytest.approx(5.0, 1.0e-6)
    assert next(result) == pytest.approx(4.0, 1.0e-6)
    assert next(result) == pytest.approx(3.0, 1.0e-6)
    assert next(result) == pytest.approx(7.0, 1.0e-6)
    assert next(result) == pytest.approx(1.0, 1.0e-6)
    assert next(result) == pytest.approx(2.0, 1.0e-6)
    assert next(result) == pytest.approx(1.0, 1.0e-6)
    assert next(result) == pytest.approx(0.0, 1.0e-6)
    assert next(result) == pytest.approx(8.0, 1.0e-6)
    with pytest.raises(StopIteration):
        next(result)


def test_state_empty_default_normalizer(setup_no_outputs):
    buffer, clock_mock = setup_no_outputs
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
    assert next(result) == pytest.approx(0.0, 1.0e-6)
    assert next(result) == pytest.approx(0.0, 1.0e-6)
    with pytest.raises(StopIteration):
        next(result)


def test_state_semifull_default_normalizer(setup_no_outputs):
    buffer, clock_mock = setup_no_outputs
    clock_mock.time = 6
    buffer.receive(Message(Header(1, 2, 3, 4)))

    result = buffer.generate_state()
    assert next(result) == pytest.approx(1.0, 1.0e-6)
    assert next(result) == pytest.approx(5.0, 1.0e-6)
    assert next(result) == pytest.approx(4.0, 1.0e-6)
    assert next(result) == pytest.approx(3.0, 1.0e-6)
    assert next(result) == pytest.approx(4.0, 1.0e-6)
    assert next(result) == pytest.approx(0.0, 1.0e-6)
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
        count_normalizer=ConstantNormalizer(0.1),
    )

    clock_mock.time = 6
    buffer.receive(Message(Header(1, 2, 3, 7)))
    buffer.receive(Message(Header(4, 5, 6, 8)))

    result = buffer.generate_state()
    assert next(result) == pytest.approx(0.5, 1.0e-6)
    assert next(result) == pytest.approx(0.5, 1.0e-6)
    assert next(result) == pytest.approx(0.4, 1.0e-6)
    assert next(result) == pytest.approx(0.3, 1.0e-6)
    assert next(result) == pytest.approx(70.0, 1.0e-6)
    assert next(result) == pytest.approx(0.5, 1.0e-6)
    assert next(result) == pytest.approx(0.2, 1.0e-6)
    assert next(result) == pytest.approx(0.1, 1.0e-6)
    assert next(result) == pytest.approx(0.0, 1.0e-6)
    assert next(result) == pytest.approx(80.0, 1.0e-6)
    with pytest.raises(StopIteration):
        next(result)
    assert len(buffer.state) == 10


def test_receive_cb_enabled(setup_outputs):
    buffer, clock_mock, output_mock, overflow_mock = setup_outputs
    buffer.set_receive_cb(lambda node: node.trigger())

    buffer.receive(Mock())
    assert output_mock.receive.call_count == 1
    assert overflow_mock.receive.call_count == 0


def test_receive_cb_disabled(setup_outputs):
    buffer, clock_mock, output_mock, overflow_mock = setup_outputs

    buffer.receive(Mock())
    assert output_mock.receive.call_count == 0
    assert overflow_mock.receive.call_count == 0
