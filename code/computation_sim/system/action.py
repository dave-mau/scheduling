from collections import namedtuple
from typing import Callable

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
