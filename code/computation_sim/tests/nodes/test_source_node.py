import pytest
from unittest.mock import Mock
from computation_sim.basic_types import CommunicationError, Message, Header
from computation_sim.nodes import source_node as sn
from computation_sim.nodes import Node
from computation_sim.time import DurationSampler


@pytest.fixture
def source_node():
    strategy_mock = Mock(spec=sn.SourceStrategy)
    return sn.SourceNode(strategy_mock, "Geralt"), strategy_mock


def test_id(source_node):
    node, _ = source_node
    assert node.id == "Geralt"


def test_receive_raises(source_node):
    node, _ = source_node
    with pytest.raises(CommunicationError):
        node.receive(Message(Header("", "", 0, 0, 0)))


def test_default_outputs_empty(source_node):
    node, _ = source_node
    assert len(node.outputs) == 0


def test_add_output(source_node):
    node, _ = source_node
    n1 = Mock(spec=Node)
    node.add_output(n1)
    n2 = Mock(spec=Node)
    node.add_output(n2)

    result = node.outputs
    assert len(node.outputs) == 2
    assert node.outputs[0] == n1
    assert node.outputs[1] == n2


def test_add_duplicate_output(source_node):
    node, _ = source_node
    n1 = Mock(spec=Node)
    node.add_output(n1)
    with pytest.raises(ValueError):
        node.add_output(n1)


def test_update_no_result(source_node):
    node, source_strategy_mock = source_node

    output_mock = Mock(spec=Node)
    node.add_output(output_mock)

    source_strategy_mock.update.return_value = None
    node.update(1)
    assert output_mock.receive.call_count == 0


def test_update_has_result(source_node):
    node, source_strategy_mock = source_node

    output_mock = Mock(spec=Node)
    node.add_output(output_mock)

    source_strategy_mock.update.return_value = "foo"
    node.update(1)
    assert output_mock.receive.call_count == 1


def test_reset_source_node(source_node):
    node, source_strategy_mock = source_node
    node.reset()
    assert source_strategy_mock.reset.call_count == 1


@pytest.fixture
def periodic_epoch_sender():
    sampler_mock = Mock(spec=DurationSampler)
    return sn.PeriodicEpochSender(4, 10, sampler_mock), sampler_mock


def test_update_no_disturbance(periodic_epoch_sender):
    sender, mock = periodic_epoch_sender
    mock.sample.return_value = 0

    assert sender.update(0) is None
    assert sender.update(4) is not None
    assert sender.update(5) is None
    assert sender.update(13) is None
    assert sender.update(14) is not None


def test_update_negative_disturbance(periodic_epoch_sender):
    sender, mock = periodic_epoch_sender
    mock.sample.side_effect = [-10, 0]

    assert sender.update(4) is not None
    assert sender.update(13) is None

    assert mock.sample.call_count == 2


def test_update_positive_disturbance(periodic_epoch_sender):
    sender, mock = periodic_epoch_sender
    mock.sample.return_value = 1

    assert sender.update(4) is not None
    assert sender.update(14) is None
    assert sender.update(15) is not None


def test_update_skip_once(periodic_epoch_sender):
    sender, mock = periodic_epoch_sender
    mock.sample.return_value = 10

    assert sender.update(4) is not None
    for i in range(20):
        assert sender.update(4 + i) is None
    assert sender.update(24) is not None


def test_reset_periodic_epoch_sender(periodic_epoch_sender):
    sender, mock = periodic_epoch_sender
    mock.sample.return_value = 1

    assert sender.update(4) is not None
    assert sender.update(4) is None
    sender.reset()
    assert sender.update(4) is not None
