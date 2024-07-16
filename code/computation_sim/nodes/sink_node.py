from typing import Generator

from computation_sim.basic_types import Message, NodeId
from computation_sim.time import TimeProvider

from .interfaces import Node, StateVariableNormalizer
from .state_normalizers import ConstantNormalizer


class SinkNode(Node):
    def __init__(
        self,
        time_provider: TimeProvider,
        id: NodeId = None,
        count_normalizer: StateVariableNormalizer = None,
    ):
        super().__init__(time_provider, id)
        self._received_messages = []
        self._receive_times = []
        self._state_normalizer = count_normalizer if count_normalizer else ConstantNormalizer(1.0)

    def receive(self, message: Message) -> None:
        self._received_messages.append(message)
        self._receive_times.append(self.time)

    def generate_state(self) -> Generator[float, None, None]:
        yield self._state_normalizer.normalize(float(len(self._received_messages)))

    def update(self):
        self._received_messages.clear()
        self._receive_times.clear()

    def trigger(self):
        pass

    def reset(self):
        self._received_messages.clear()
        self._receive_times.clear()
