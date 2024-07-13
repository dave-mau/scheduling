from unittest.mock import Mock

import pytest
from computation_sim.basic_types import CommunicationError, Header, Message
from computation_sim.nodes import Node
from computation_sim.nodes import source_node as sn
from computation_sim.time import DurationSampler, TimeProvider


@pytest.fixture
def setup():
    sensor_mock = Mock(spec=sn.Sensor)
    clock_mock = Mock(sepc=TimeProvider)
    return sn.SourceNode(clock_mock, sensor_mock, "Geralt"), sensor_mock, clock_mock


def test_id(setup):
    node, _, _ = setup
    assert node.id == "Geralt"


def test_receive_raises(setup):
    node, _, _ = setup
    with pytest.raises(CommunicationError):
        node.receive(Message(Header()))


def test_default_outputs_empty(setup):
    node, _, _ = setup
    assert len(node.outputs) == 0


def test_setup_add_output(setup):
    node, _, _ = setup
    n1 = Mock(spec=Node)
    node.add_output(n1)
    n2 = Mock(spec=Node)
    node.add_output(n2)

    result = node.outputs
    assert len(node.outputs) == 2
    assert node.outputs[0] == n1
    assert node.outputs[1] == n2


def test_add_duplicate_output(setup):
    node, _, _ = setup
    n1 = Mock(spec=Node)
    node.add_output(n1)
    with pytest.raises(ValueError):
        node.add_output(n1)


def test_update_sensor_has_no_measurement(setup):
    node, sensor_mock, clock_mock = setup

    output_mock = Mock(spec=Node)
    node.add_output(output_mock)

    sensor_mock.get_measurement.return_value = None
    clock_mock.time.return_value = 1
    node.update()
    assert output_mock.receive.call_count == 0


def test_update_sensors_has_measurement(setup):
    node, sensor_mock, clock_mock = setup

    output_mock = Mock(spec=Node)
    node.add_output(output_mock)

    sensor_mock.get_measurement.return_value = Mock()
    clock_mock.time.return_value = 1
    node.update()

    assert output_mock.receive.call_count == 1
    assert sensor_mock.update.call_count == 1


def test_reset_source_node(setup):
    node, sensor_mock, clock_mock = setup
    node.reset()
    assert sensor_mock.reset.call_count == 1
