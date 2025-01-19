from copy import deepcopy
from typing import Callable, Generator, Optional

from computation_sim.basic_types import Message, NodeId
from computation_sim.time import TimeProvider

from .interfaces import Node, StateVariableNormalizer
from .state_normalizers import ConstantNormalizer
from .utils import empty_message_state, header_to_state


class OutputNode(Node):
    def __init__(
        self,
        time_provider: TimeProvider,
        id: NodeId = None,
        receive_cb: Callable[[Message], None] = None,
        age_normalizer: StateVariableNormalizer = None,
        occupancy_normalizer: StateVariableNormalizer = None,
        count_normalizer: StateVariableNormalizer = None,
    ):
        super().__init__(time_provider, id)
        self._receive_cb = receive_cb
        self._last_receive_time = None
        self._last_received = None
        self._age_normalizer = age_normalizer if age_normalizer else ConstantNormalizer(1.0)
        self._occupancy_normalizer = occupancy_normalizer if occupancy_normalizer else ConstantNormalizer(1.0)
        self._count_normalizer = count_normalizer if count_normalizer else ConstantNormalizer(1.0)

    def receive(self, message: Message) -> None:
        self._last_received = deepcopy(message)
        self._last_receive_time = self.time
        if self._receive_cb:
            self._receive_cb(message)

    def generate_state(self) -> Generator[float, None, None]:
        if self._last_received:
            yield self._occupancy_normalizer.normalize(1.0)
            state = header_to_state(
                self._last_received.header,
                self.time,
                age_normalizer=self._age_normalizer,
                count_normalizer=self._count_normalizer,
            )
        else:
            yield self._occupancy_normalizer.normalize(0.0)
            state = empty_message_state(age_normalizer=self._age_normalizer, count_normalizer=self._count_normalizer)

        for val in state:
            yield val

    @property
    def draw_options(self) -> dict:
        state = self.state

        color = "dimgrey"
        if self._last_received:
            color = "floralwhite" if self._last_receive_time == self.time else "lightgrey"
        return dict(
            color=color,
            symbol="circle",
            hovertext=f"is_occupied = {state[0]}<br>msg.age_oldest = {state[1]}<br>msg.age_youngest = {state[2]}<br>msg.age_average = {state[3]}<br>msg.num_measurements = {state[4]}<br>",
        )

    @property
    def last_received(self) -> Optional[Message]:
        return self._last_received

    def update(self):
        # nothing to do
        pass

    def trigger(self):
        # nothing to do
        pass

    def reset(self):
        self._last_received = None
