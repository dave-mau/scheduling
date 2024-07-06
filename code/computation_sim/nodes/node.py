from abc import ABC, abstractmethod
from typing import List, Dict
from uuid import uuid4
from computation_sim.basic_types import Message, NodeId, Time
from computation_sim.time import TimeProvider


class Visitor:
    @abstractmethod
    def visit_node(self, node: "Node"):
        pass


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

    @property
    @abstractmethod
    def state(self) -> List[float]:
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def trigger(self):
        pass

    # TODO: Add "trigger"

    # @abstractmethod
    # def draw(self, draw_context: DrawContext):
    #    pass

    def visit(self, visitor: Visitor):
        visitor.visit_node(self)

    @abstractmethod
    def reset(self):
        pass
