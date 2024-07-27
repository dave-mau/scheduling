import random
from collections import deque, namedtuple

import numpy as np
import torch

Sample = namedtuple("Sample", ("s", "a", "s_prime", "r"))


class Memory(object):
    def __init__(self, capacity: int):
        self._memory = deque([], capacity)

    def push(self, *args):
        self._memory.append(Sample(*args))

    def sample(self, batch_size):
        return random.sample(self._memory, batch_size)

    def __contains__(self, val) -> bool:
        return val in self._memory

    def __len__(self):
        return len(self._memory)
