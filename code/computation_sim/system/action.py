from collections import namedtuple
from math import floor, log2
from typing import Callable

import numpy as np

ActionCallback = namedtuple("ActionCallback", ("callback", "priority", "name"))


class Action(object):
    def __init__(self, name: str = ""):
        self.name = name
        self._callbacks = []
        self._readiness_callback = lambda: True

    def __len__(self) -> int:
        return len(self._callbacks)

    def __repr__(self) -> str:
        return f'Action "{self.name}" with {len(self)} callbacks.'

    def register_readiness_callback(self, callback: Callable[[], bool]) -> None:
        self._readiness_callback = callback

    def register_callback(self, callback: Callable, priority: int, name: str = ""):
        self._callbacks.append(ActionCallback(callback, priority, name))
        self._callbacks.sort(key=lambda x: x.priority, reverse=True)

    def act(self) -> None:
        if not self._readiness_callback():
            return
        for cb in self._callbacks:
            cb.callback()

    def clear(self) -> None:
        self._callbacks.clear()


def max_action_id(num_action_dims: int) -> int:
    """Returns the largest possible action id, given the size of the action vector.

    For example, given a 3D action vector, the largest action vector is ([1, 1, 1]), which is 7 in decimal.
    """
    return 2**num_action_dims - 1


def num_actions(num_action_dims: int) -> int:
    """Returns the number of possible actions, given the size of the action vector.

    For example, given a three dimensional action vector, there are 8 possible combinations.
    """
    if num_action_dims > 0:
        return max_action_id(num_action_dims) + 1
    else:
        return 0


def unpack_action(num_action_dims: int, action: int) -> np.ndarray:
    """Given a a packed action `action`, convert it into its unpacked form.

    Example:
        ```
        assert unpack_action(3, 1) == [0, 0, 1]
        assert unpack_action(3, 2) == [0, 1, 0]
        assert unpack_action(3, 7) == [1, 1, 1]
    """
    num_actions = max_action_id(num_action_dims)
    assert action >= 0 and action <= num_actions, f"Invalid action id {action}."
    assert 0 < num_actions <= 255, f"Maximum number of actions is 255."
    bits = np.unpackbits(np.uint8(action))
    return bits[-num_action_dims:]
