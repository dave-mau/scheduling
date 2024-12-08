import pytest
from computation_sim.time.duration_samplers import (
    FixedDuration,
    GammaDistributionSampler,
    GaussianTimeSampler,
)


def test_fixed_duration():
    sampler = FixedDuration(10)
    assert sampler.sample() == 10
    assert sampler.sample() == 10
    assert sampler.sample() == 10
    assert sampler.sample() == 10


def test_gaussian():
    sampler = GaussianTimeSampler(0.0, 1.0, 1.0, 1.0)
    sampler.reset(0)
    data_0 = [sampler.sample() for _ in range(10)]

    sampler.reset(0)
    data_1 = [sampler.sample() for _ in range(10)]

    assert data_0 == data_1


def test_gamma():
    sampler = GammaDistributionSampler(1.0, 2.0)
    sampler.reset(0)
    data_0 = [sampler.sample() for _ in range(10)]

    sampler.reset(0)
    data_1 = [sampler.sample() for _ in range(10)]

    assert data_0 == data_1
