from abc import ABC, abstractmethod

import numpy as np

from .clock import TimeMs, round_to_milliseconds


class ExecutionTimeSampler(ABC):
    """Base class of all execution time samplers.

    An execution time sampler models a probability distibution over the time
    it takes to complete a process. This base class serves as a common interface
    to all the execution time samplers.
    """

    @abstractmethod
    def sample(self) -> TimeMs:
        pass


class GaussianTimeSampler(ExecutionTimeSampler):
    def __init__(self, mu: float, std: float, t_min: float = None, t_max: float = None):
        """Execution times that are distributed according to a normal distribution.

        Args:
            mu (_type_): Mean of the gaussian.
            std (_type_): Standard deviation.
            t_min (float, optional): Optional value to clip values at lower bound. Defaults to None.
            t_max (float, optional): Optional value to clip values at upper bound. Defaults to None.
        """
        self._mu = mu
        self._std = std
        self._t_min = t_min
        self._t_max = t_max

    @round_to_milliseconds
    def sample(self) -> TimeMs:
        if self._t_min == None and self._t_max == None:
            return np.random.normal(self._mu, self._std)
        return np.clip(np.random.normal(self._mu, self._std), self._t_min, self._t_max)


class GammaDistributionSampler(ExecutionTimeSampler):
    def __init__(self, k, theta, gain: float = 1.0, offset: float = 0.0):
        """Execution times that are distributed according to a gamma distributtion.

        This class samples values X from a gamma distribution with parameters (k, theta).
        In addition, samples are scaled and offsetted: Y = gain * X + offset. The resulting values
        are rounded to the nearest integer (time in ms).

        Args:
            k (float): Shape parameter.
            theta (float): Scale parameter.
            gain (float, optional): . Defaults to 1.0.
            offset (float, optional): _description_. Defaults to 63.0.
        """
        self._k = k
        self._theta = theta
        self._gain = gain
        self._offset = offset
        pass

    @round_to_milliseconds
    def sample(self) -> TimeMs:
        return np.random.gamma(self._k, self._theta) * self._gain + self._offset


class FixedTime(ExecutionTimeSampler):
    """A fixed execution time that does not have any stochasticity."""

    def __init__(self, val: TimeMs):
        self._val = val

    @round_to_milliseconds
    def sample(self):
        return self._val
