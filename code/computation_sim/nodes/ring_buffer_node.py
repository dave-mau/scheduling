from collections import deque
from typing import Callable, Generator, Optional

from computation_sim.basic_types import Message, NodeId
from computation_sim.time import TimeProvider

from .interfaces import Node, StateVariableNormalizer
from .state_normalizers import ConstantNormalizer
from .utils import header_to_state


class RingBufferNode(Node):
    def __init__(
        self,
        time_provider: TimeProvider,
        id: NodeId = None,
        max_num_elements=1,
        overflow_cb: Callable[[Message], None] = None,
        age_normalizer: StateVariableNormalizer = None,
        occupancy_normalizer: StateVariableNormalizer = None,
    ):
        super().__init__(time_provider, id)
        self._buffer = deque(maxlen=max_num_elements)
        self._overflow_cb = overflow_cb
        self._age_normalizer = age_normalizer if age_normalizer else ConstantNormalizer(1.0)
        self._occupancy_normalizer = occupancy_normalizer if occupancy_normalizer else ConstantNormalizer(1.0)

    def receive(self, message: Message) -> None:
        if self._overflow_cb and (self.num_entries == self._buffer.maxlen):
            self._overflow_cb(self._buffer.popleft())
        self._buffer.append(message)

    def generate_state(self) -> Generator[float, None, None]:
        # Write non-empty elements
        for element in self._buffer:
            yield self._occupancy_normalizer.normalize(1.0)
            for x in header_to_state(element.header, self.time):
                yield self._age_normalizer.normalize(x)
        # Write empty elements
        for _ in range(self.maxlen - len(self._buffer)):
            yield self._occupancy_normalizer.normalize(0.0)
            yield self._age_normalizer.normalize(0.0)
            yield self._age_normalizer.normalize(0.0)
            yield self._age_normalizer.normalize(0.0)

    @property
    def draw_options(self) -> dict:
        if self.num_entries == self.maxlen:
            color = "darkred"
        elif self.num_entries == 0:
            color = "darkgreen"
        else:
            color = "darkorange"
        hovertext=""
        state = self.state
        for occupied, oldest, youngest, average in zip(state[0::4], state[1::4], state[2::4], state[3::4]):
            hovertext += f"is_occupied = {occupied}<br>msg.age_oldest = {oldest}<br>msg.age_youngest = {youngest}<br>msg.age_average = {average}<br>"

        return dict(
            color=color,
            symbol="square",
            hovertext=hovertext,
        )

    @property
    def num_entries(self) -> int:
        return len(self._buffer)

    @property
    def maxlen(self) -> int:
        return self._buffer.maxlen

    def update(self):
        pass

    def trigger(self):
        if self.num_entries > 0:
            message = self._buffer.popleft()
            for output in self._outputs:
                output.receive(message)

    def reset(self):
        self._buffer.clear()

    def add_output(self, output: Node):
        self._outputs.append(output)

    def pop(self) -> Optional[Message]:
        return self._buffer.popleft() if len(self._buffer) > 0 else None
