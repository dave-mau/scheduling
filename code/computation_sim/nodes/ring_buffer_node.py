from .node import Node
from .utils import header_to_state
from computation_sim.time import TimeProvider
from computation_sim.basic_types import Message, NodeId, Time
from typing import List, Callable, Optional
from collections import deque


class RingBufferNode(Node):
    def __init__(
        self,
        time_provider: TimeProvider,
        id: NodeId = None,
        max_num_elements=1,
        overflow_cb: Callable[[Message], None] = None,
    ):
        super().__init__(time_provider, id)
        self._buffer = deque(maxlen=max_num_elements)
        self._overflow_cb = overflow_cb

    def receive(self, message: Message) -> None:
        if self._overflow_cb and (self.num_entries == self._buffer.maxlen):
            self._overflow_cb(self._buffer.popleft())
        self._buffer.append(message)

    @property
    def state(self) -> List[float]:
        result = []
        for element in self._buffer:
            result.extend(header_to_state(element.header, self.time))

    @property
    def num_entries(self) -> int:
        return len(self._buffer)

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
