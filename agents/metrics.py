import numpy as np


class MovingTotal:
    def __init__(self, maxlen):
        self.data = np.zeros((maxlen,))
        self._total = 0
        self._push = 0
        self._len = 0

    def push(self, val):
        if self._len == len(self.data):
            self._total -= self.data[self._push]
        self._total += val

        self.data[self._push] = val
        self._len = min(self._len + 1, len(self.data))
        self._push = self._advance(self._push)

    def _advance(self, it):
        return (it + 1) % len(self.data)

    @property
    def value(self):
        return self._total


class MovingAverage(MovingTotal):
    @property
    def value(self):
        return self._total / max(self._len, 1)
