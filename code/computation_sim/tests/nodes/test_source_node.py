import pytest
from unittest.mock import Mock
from computation_sim.basic_types import CommunicationError, Message, Header
from computation_sim.nodes import source_node as sn
from computation_sim.nodes import Node
from computation_sim.time import DurationSampler


@pytest.fixture
def setup():
    sensor_mock = Mock(spec=sn.Sensor)
    return sn.SourceNode(sensor_mock, "Geralt"), sensor_mock


def test_id(setup):
    node, _ = setup
    assert node.id == "Geralt"


def test_receive_raises(setup):
    node, _ = setup
    with pytest.raises(CommunicationError):
        node.receive(Message(Header()))


def test_default_outputs_empty(setup):
    node, _ = setup
    assert len(node.outputs) == 0


def test_setup_add_output(setup):
    node, _ = setup
    n1 = Mock(spec=Node)
    node.add_output(n1)
    n2 = Mock(spec=Node)
    node.add_output(n2)

    result = node.outputs
    assert len(node.outputs) == 2
    assert node.outputs[0] == n1
    assert node.outputs[1] == n2


def test_add_duplicate_output(setup):
    node, _ = setup
    n1 = Mock(spec=Node)
    node.add_output(n1)
    with pytest.raises(ValueError):
        node.add_output(n1)


def test_update_sensor_has_no_measurement(setup):
    node, sensor_mock = setup

    output_mock = Mock(spec=Node)
    node.add_output(output_mock)

    sensor_mock.get_measurement.return_value = None
    node.update(1)
    assert output_mock.receive.call_count == 0


def test_update_sensors_has_measurement(setup):
    node, sensor_mock = setup

    output_mock = Mock(spec=Node)
    node.add_output(output_mock)

    sensor_mock.get_measurement.return_value = Mock()
    node.update(1)
    assert output_mock.receive.call_count == 1
    assert sensor_mock.update.call_count == 1


def test_reset_source_node(setup):
    node, sensor_mock = setup
    node.reset()
    assert sensor_mock.reset.call_count == 1
