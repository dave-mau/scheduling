from ..basic_types import NodeId, Message, CommunicationError, Time
from . import Node
from abc import ABC, abstractmethod
from typing import List, Optional


class SourceStrategy(ABC):
    @abstractmethod
    def update(self, time: Time) -> Optional[Message]:
        pass

    @abstractmethod
    def reset(self):
        pass


class SourceNode(Node):
    def __init__(self, strategy: SourceStrategy, id: NodeId = None):
        super().__init__(id)
        self._strategy = strategy
        self._last_send_time = None

    def receive(self, message: Message) -> None:
        super().receive(message)
        raise CommunicationError("Source node cannot receive a message.")

    def add_output(self, output: Node) -> None:
        super().add_output(output)
        self._outputs[output.id] = output

    @property
    def state(self) -> List[float]:
        return [self._last_send_time]

    def update(self, time: Time):
        result: Message = self._strategy.update(time)
        if result:
            self._last_send_time = time
            self._outputs[result.destination_id].receive(result)

    def reset(self):
        self._strategy.reset()
        self._last_send_time = None


class PeriodicSender(SourceStrategy):
    pass


#    def __init__()
