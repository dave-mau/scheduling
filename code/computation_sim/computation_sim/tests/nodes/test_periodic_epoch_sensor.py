from unittest.mock import Mock

import computation_sim.nodes as sn
import pytest
from computation_sim.time import DurationSampler


@pytest.fixture
def setup():
    sampler_mock = Mock(spec=DurationSampler)
    return sn.PeriodicEpochSensor(4, 10, sampler_mock), sampler_mock


def test_state(setup):
    sensor, _ = setup
    assert sensor.state == []


def test_state_generator(setup):
    sensor, _ = setup
    result = sensor.generate_state()
    with pytest.raises(StopIteration):
        next(result)


def test_update_no_disturbance(setup):
    sensor, mock_sampler = setup
    mock_sampler.sample.return_value = 0

    sensor.update(0)
    assert not sensor.has_measurement

    sensor.update(4)
    assert sensor.has_measurement
    sensor.update(5)
    assert not sensor.has_measurement
    sensor.update(13)
    assert not sensor.has_measurement
    sensor.update(14)
    assert sensor.has_measurement


def test_update_negative_disturbance(setup):
    sensor, mock_sampler = setup
    mock_sampler.sample.side_effect = [-10, 0]

    sensor.update(4)
    assert sensor.has_measurement
    sensor.update(13)
    assert not sensor.has_measurement

    assert mock_sampler.sample.call_count == 2


def test_update_positive_disturbance(setup):
    sensor, mock_sampler = setup
    mock_sampler.sample.return_value = 1

    sensor.update(4)
    assert sensor.has_measurement
    sensor.update(14)
    assert not sensor.has_measurement
    sensor.update(15)
    assert sensor.has_measurement


def test_update_skip_once(setup):
    sensor, mock_sampler = setup
    mock_sampler.sample.return_value = 10

    sensor.update(4)
    assert sensor.has_measurement
    for i in range(20):
        sensor.update(4 + i)
        assert not sensor.has_measurement
    sensor.update(24)
    assert sensor.has_measurement


def test_reset(setup):
    sensor, mock_sampler = setup
    mock_sampler.sample.return_value = 1

    sensor.update(4)
    assert sensor.has_measurement
    sensor.update(4)
    assert not sensor.has_measurement
    sensor.reset()
    sensor.update(4)
    assert sensor.has_measurement


def test_get_measurement(setup):
    sensor, mock_sampler = setup
    mock_sampler.sample.return_value = 0

    assert not sensor.get_measurement()
    sensor.update(4)
    result = sensor.get_measurement()

    # Advance clock, to make sure that message is copied
    sensor.update(10)
    assert result
    assert result.header.t_measure_average == 4
    assert result.header.t_measure_oldest == 4
    assert result.header.t_measure_youngest == 4
