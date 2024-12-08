from abc import ABC, abstractmethod

import numpy as np
from computation_sim.basic_types import Time

from .clock import round_to_fixed_point


class DurationSampler(ABC):
    def __init__(self, seed: int = None):
        self._rng = np.random.default_rng(seed)

    @property
    def rng(self) -> np.random.Generator:
        return self

    def reset(self, seed: int = None):
        self._rng = np.random.default_rng(seed)

    @abstractmethod
    def sample(self) -> Time:
        pass


class FixedDuration(DurationSampler):
    def __init__(self, val: Time, **kwargs):
        super().__init__(**kwargs)
        self.val = val

    @round_to_fixed_point
    def sample(self):
        return self.val


class GaussianTimeSampler(DurationSampler):
    def __init__(self, mu: float, std: float, gain: float, offset: float, **kwargs):
        super().__init__(**kwargs)
        self._mu = mu
        self._std = std
        self._gain = gain
        self._offset = offset

    @round_to_fixed_point
    def sample(self) -> Time:
        res = -1
        while res < 0:
            res = self._rng.normal(self._mu, self._std) * self._gain + self._offset
        return res


class GammaDistributionSampler(DurationSampler):
    def __init__(self, k, theta, gain: float = 1.0, offset: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self._k = k
        self._theta = theta
        self._gain = gain
        self._offset = offset

    @round_to_fixed_point
    def sample(self) -> Time:
        return self._rng.gamma(self._k, self._theta) * self._gain + self._offset
