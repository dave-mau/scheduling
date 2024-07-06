from .node import Node
from .utils import header_to_state
from computation_sim.basic_types import Message, NodeId, Time
from computation_sim.time import as_age
from typing import List, Callable
from copy import deepcopy


class SinkNode(Node):
    def __init__(self, id: NodeId = None):
        super().__init__(id)
        self._received_messages = []

    def receive(self, message: Message) -> None:
        self._received_messages.append(message)

    @property
    def state(self) -> List[float]:
        return [float(len(self._received_messages))]

    def update(self, time: Time):
        self._received_messages.clear()

    def trigger(self):
        pass

    def reset(self):
        self._received_messages.clear()
