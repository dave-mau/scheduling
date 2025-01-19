from abc import ABC, abstractmethod
from typing import Generator, List, Optional
from uuid import uuid4

from computation_sim.basic_types import Message, NodeId, Time
from computation_sim.time import TimeProvider


class Node(ABC):
    def __init__(self, time_provider: TimeProvider, id: NodeId = None):
        self._time_provider = time_provider
        self.__id = id if id else uuid4()
        self._outputs: List[Node] = []

    @property
    def id(self) -> NodeId:
        return self.__id

    @property
    def outputs(self) -> List["Node"]:
        return self._outputs

    @property
    def time(self) -> Time:
        return self._time_provider.time

    @abstractmethod
    def receive(self, message: Message) -> None:
        pass

    @abstractmethod
    def generate_state(self) -> Generator[float, None, None]:
        pass

    @property
    def state(self) -> List[float]:
        return list(self.generate_state())

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def trigger(self):
        pass

    @property
    def draw_options(self) -> dict:
        return {}

    # def draw(self, draw_context: DrawContext):
    #    pass

    def visit(self, visitor: "NodeVisitor"):
        visitor.visit_node(self)

    @abstractmethod
    def reset(self):
        pass


class NodeVisitor(ABC):
    @abstractmethod
    def visit_node(self, node: "Node"):
        pass


class Sensor(ABC):
    def __init__(self):
        self._last_update_time: Optional[Time] = None
        self._has_measurement = False

    @abstractmethod
    def generate_state(self) -> Generator[float, None, None]:
        pass

    @property
    def state(self) -> List[float]:
        return list(self.generate_state())

    @property
    def has_measurement(self) -> bool:
        return self._has_measurement

    @abstractmethod
    def get_measurement(self) -> Optional[Message]:
        pass

    def update(self, time: Time):
        self._last_update_time = time

    def reset(self) -> None:
        self._last_update_time = None
        self._has_measurement = False


class StateVariableNormalizer(ABC):
    @abstractmethod
    def normalize(self, value: float) -> float:
        pass
