from copy import deepcopy
from typing import Callable, Generator, Optional

from computation_sim.basic_types import Message, NodeId
from computation_sim.time import TimeProvider

from .interfaces import Node, StateVariableNormalizer
from .utils import header_to_state
from .state_normalizers import ConstantNormalizer


class OutputNode(Node):
    def __init__(
        self,
        time_provider: TimeProvider,
        id: NodeId = None,
        receive_cb: Callable[[Message], None] = None,
        age_normalizer: StateVariableNormalizer = None,
        occupancy_normalizer: StateVariableNormalizer = None,
    ):
        super().__init__(time_provider, id)
        self._receive_cb = receive_cb
        self._last_received = None
        self._age_normalizer = (
            age_normalizer if age_normalizer else ConstantNormalizer(1.0)
        )
        self._occupancy_normalizer = (
            occupancy_normalizer if occupancy_normalizer else ConstantNormalizer(1.0)
        )

    def receive(self, message: Message) -> None:
        self._last_received = deepcopy(message)
        if self._receive_cb:
            self._receive_cb(message)

    def generate_state(self) -> Generator[float, None, None]:
        if self._last_received:
            vals = (1.0, *header_to_state(self._last_received.header, self.time))
        else:
            vals = (0.0, 0.0, 0.0, 0.0)
        yield self._occupancy_normalizer.normalize(vals[0])
        yield self._age_normalizer.normalize(vals[1])
        yield self._age_normalizer.normalize(vals[2])
        yield self._age_normalizer.normalize(vals[3])

    @property
    def last_received(self) -> Optional[Message]:
        return self._last_received

    def update(self):
        pass

    def trigger(self):
        pass

    def reset(self):
        self._last_received = None
