from abc import ABC, abstractmethod
from typing import List
from uuid import uuid4
from .types import Time, NodeId
from .message import Message
from .draw_context import DrawContext


class Visitor:
    @abstractmethod
    def visit_node(self, node: "Node"):
        pass


class Node(ABC):
    def __init__(self, id: NodeId = None):
        self.__id = id if id else uuid4()

    @property
    def id(self) -> NodeId:
        return self.__id

    @abstractmethod
    def receive(self, message: Message):
        pass

    @property
    @abstractmethod
    def outputs(self) -> List["Node"]:
        pass

    @abstractmethod
    def add_output(self, output: "Node") -> None:
        pass

    @property
    @abstractmethod
    def state(self) -> List[float]:
        pass

    @abstractmethod
    def update(self, time: Time):
        pass

    @abstractmethod
    def draw(self, draw_context: DrawContext):
        pass

    def visit(self, visitor: Visitor):
        visitor.visit_node(self)
