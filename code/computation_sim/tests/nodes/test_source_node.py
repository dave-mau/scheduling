import pytest
from unittest.mock import Mock
from computation_sim.nodes import source_node as sn
from computation_sim.time import DurationSampler


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


def test_reset(periodic_epoch_sender):
    sender, mock = periodic_epoch_sender
    mock.sample.return_value = 1

    assert sender.update(4) is not None
    assert sender.update(4) is None
    sender.reset()
    assert sender.update(4) is not None
