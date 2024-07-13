from .interfaces import Node
from .utils import header_to_state
from computation_sim.time import TimeProvider
from computation_sim.basic_types import Message, NodeId
from typing import List, Callable, Optional
from copy import deepcopy


class OutputNode(Node):
    def __init__(
        self,
        time_provider: TimeProvider,
        id: NodeId = None,
        receive_cb: Callable[[Message], None] = None,
    ):
        super().__init__(time_provider, id)
        self._receive_cb = receive_cb
        self._last_received = None

    def receive(self, message: Message) -> None:
        self._last_received = deepcopy(message)
        if self._receive_cb:
            self._receive_cb(message)

    @property
    def state(self) -> List[float]:
        if self._last_received:
            return [1.0, *header_to_state(self._last_received.header, self.time)]
        else:
            return 4 * [0.0]

    @property
    def last_received(self) -> Optional[Message]:
        return self._last_received

    def update(self):
        pass

    def trigger(self):
        pass

    def reset(self):
        self._last_received = None
