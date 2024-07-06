from .node import Node
from .utils import header_to_state
from computation_sim.basic_types import Message, NodeId, Time
from computation_sim.time import as_age, TimeProvider
from typing import List, Callable
from copy import deepcopy


class SinkNode(Node):
    def __init__(self, time_provider: TimeProvider, id: NodeId = None):
        super().__init__(time_provider, id)
        self._received_messages = []
        self._receive_times = []

    def receive(self, message: Message) -> None:
        self._received_messages.append(message)
        self._receive_times.append(self.time)

    @property
    def state(self) -> List[float]:
        return [float(len(self._received_messages))]

    def update(self, time: Time):
        self._received_messages.clear()
        self._receive_times.clear()

    def trigger(self):
        pass

    def reset(self):
        self._received_messages.clear()
        self._receive_times.clear()
