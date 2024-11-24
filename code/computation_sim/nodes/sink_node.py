from typing import Generator, List

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

    @property
    def count(self) -> int:
        return len(self.received_messages)

    @property
    def received_messages(self) -> List[Message]:
        return self._received_messages

    @property
    def received_times(self) -> List[Message]:
        return self._receive_times

    def receive(self, message: Message) -> None:
        self._received_messages.append(message)
        self._receive_times.append(self.time)

    def generate_state(self) -> Generator[float, None, None]:
        yield self._state_normalizer.normalize(float(len(self._received_messages)))

    @property
    def draw_options(self) -> dict:
        color = "darkgrey" if len(self._received_messages) == 0 else "dimgrey"
        return dict(color=color, symbol="triangle-down", hovertext=f"num_messages = {len(self._received_messages)}")

    def update(self):
        pass

    def trigger(self):
        pass

    def reset(self):
        self._received_messages.clear()
        self._receive_times.clear()
