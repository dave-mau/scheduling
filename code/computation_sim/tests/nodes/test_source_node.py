import pytest
from unittest.mock import Mock
from computation_sim.basic_types import CommunicationError, Message, Header
from computation_sim.nodes import source_node as sn
from computation_sim.nodes import Node
from computation_sim.time import DurationSampler


@pytest.fixture
def source_node():
    trigger_mock = Mock(spec=sn.SourceTrigger)
    return sn.SourceNode(trigger_mock, "Geralt"), trigger_mock


def test_source_node_id(source_node):
    node, _ = source_node
    assert node.id == "Geralt"


def test_source_node_receive_raises(source_node):
    node, _ = source_node
    with pytest.raises(CommunicationError):
        node.receive(Message(Header("", "", 0, 0, 0)))


def test_source_node_default_outputs_empty(source_node):
    node, _ = source_node
    assert len(node.outputs) == 0


def test_source_node_add_output(source_node):
    node, _ = source_node
    n1 = Mock(spec=Node)
    node.add_output(n1)
    n2 = Mock(spec=Node)
    node.add_output(n2)

    result = node.outputs
    assert len(node.outputs) == 2
    assert node.outputs[0] == n1
    assert node.outputs[1] == n2


def test_source_node_add_duplicate_output(source_node):
    node, _ = source_node
    n1 = Mock(spec=Node)
    node.add_output(n1)
    with pytest.raises(ValueError):
        node.add_output(n1)


def test_source_node_update_no_result(source_node):
    node, trigger_mock = source_node

    output_mock = Mock(spec=Node)
    node.add_output(output_mock)

    trigger_mock.update.return_value = None
    node.update(1)
    assert output_mock.receive.call_count == 0


def test_source_node_update_has_result(source_node):
    node, trigger_mock = source_node

    output_mock = Mock(spec=Node)
    node.add_output(output_mock)

    trigger_mock.update.return_value = "foo"
    node.update(1)
    assert output_mock.receive.call_count == 1
    assert trigger_mock.update.call_count == 1


def test_source_node_reset_source_node(source_node):
    node, trigger_mock = source_node
    node.reset()
    assert trigger_mock.reset.call_count == 1


@pytest.fixture
def periodic_epoch_trigger():
    sampler_mock = Mock(spec=DurationSampler)
    return sn.PeriodicEpochTrigger(4, 10, sampler_mock), sampler_mock


def test_periodic_epoch_trigger_update_no_disturbance(periodic_epoch_trigger):
    trigger, mock = periodic_epoch_trigger
    mock.sample.return_value = 0

    assert trigger.update(0) is None
    assert trigger.update(4) is not None
    assert trigger.update(5) is None
    assert trigger.update(13) is None
    assert trigger.update(14) is not None


def test_periodic_epoch_trigger_update_negative_disturbance(periodic_epoch_trigger):
    trigger, mock = periodic_epoch_trigger
    mock.sample.side_effect = [-10, 0]

    assert trigger.update(4) is not None
    assert trigger.update(13) is None

    assert mock.sample.call_count == 2


def test_periodic_epoch_trigger_update_positive_disturbance(periodic_epoch_trigger):
    trigger, mock = periodic_epoch_trigger
    mock.sample.return_value = 1

    assert trigger.update(4) is not None
    assert trigger.update(14) is None
    assert trigger.update(15) is not None


def test_periodic_epoch_trigger_update_skip_once(periodic_epoch_trigger):
    trigger, mock = periodic_epoch_trigger
    mock.sample.return_value = 10

    assert trigger.update(4) is not None
    for i in range(20):
        assert trigger.update(4 + i) is None
    assert trigger.update(24) is not None


def test_periodic_epoch_trigger_reset_periodic_epoch_trigger(periodic_epoch_trigger):
    trigger, mock = periodic_epoch_trigger
    mock.sample.return_value = 1

    assert trigger.update(4) is not None
    assert trigger.update(4) is None
    trigger.reset()
    assert trigger.update(4) is not None


def test_periodic_epoch_trigger_mock_sensor():
    sampler_mock = Mock(spec=DurationSampler)
    sensor_mock = Mock(spec=sn.Sensor)

    sampler_mock.sample.return_value = 0
    sensor_mock.measure.return_value = "A beautiful picture"

    trigger = sn.PeriodicEpochTrigger(0, 10, sampler_mock, sensor=sensor_mock)
    result = trigger.update(0)

    assert sampler_mock.sample.call_count == 1
    assert sensor_mock.measure.call_count == 1
    assert result == "A beautiful picture"
