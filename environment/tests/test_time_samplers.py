import pytest
from environment import GaussianTimeSampler, GammaDistributionSampler, TimeMs
import numpy as np


@pytest.fixture
def gaussian() -> GaussianTimeSampler:
    return GaussianTimeSampler(10, 2, -1, 15)


def test_gaussian_sample(gaussian):
    result = gaussian.sample()
    assert result is not None
    assert type(result) == TimeMs


def test_gaussian_samples_within_bounds(gaussian):
    vals = np.array([gaussian.sample() for _ in range(10000)])
    assert not np.any(vals < -1)
    assert not np.any(vals > 15)


@pytest.fixture
def gamma():
    return GammaDistributionSampler(1, 2, 0.5, 1)


def test_gamma_sample(gamma):
    result = gamma.sample()
    assert result is not None
    assert type(result) == TimeMs
