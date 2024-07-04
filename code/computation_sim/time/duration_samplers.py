import numpy as np
from abc import ABC, abstractmethod
from ..basic_types import Time
from .clock import round_to_fixed_point


class DurationSampler(ABC):
    @abstractmethod
    def sample(self) -> Time:
        pass


class FixedDuration(DurationSampler):
    def __init__(self, val: Time):
        self.val = val

    @round_to_fixed_point
    def sample(self):
        return self.val


class GaussianTimeSampler(DurationSampler):
    def __init__(self, mu: float, std: float):
        self._mu = mu
        self._std = std

    @round_to_fixed_point
    def sample(self) -> Time:
        return np.random.normal(self._mu, self._std)


class GammaDistributionSampler(DurationSampler):
    def __init__(self, k, theta, gain: float = 1.0, offset: float = 0.0):
        self._k = k
        self._theta = theta
        self._gain = gain
        self._offset = offset

    @round_to_fixed_point
    def sample(self) -> Time:
        return np.random.gamma(self._k, self._theta) * self._gain + self._offset
