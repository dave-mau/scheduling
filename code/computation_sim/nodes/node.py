from abc import ABC, abstractmethod
from typing import List
from uuid import uuid4
from .types import Time, NodeId
from .message import Message


class Visitor(ABC):
    @abstractmethod
    def visit_node(self, node: "Node"):
        pass


class Node(ABC):
    def __init__(self, id: NodeId = None):
        self.__id = id if id else uuid4()

    @property
    def id(self) -> NodeId:
        return self.__id

    @property
    @abstractmethod
    def inputs(self) -> List["Node"]:
        pass

    @property
    @abstractmethod
    def outputs(self) -> List["Node"]:
        pass

    @property
    @abstractmethod
    def state(self) -> List[float]:
        """
        Describe what this property represents.
        """
        pass

    @abstractmethod
    def update(self, time: Time):
        """
        Update the node's state based on the given time.
        """
        pass

    def visit(self, visitor: Visitor):
        """
        Accept a visitor and let it visit this node.
        """
        visitor.visit_node(self)

    @abstractmethod
    def receive(self, message: Message):
        """
        Receive a message and process it accordingly.
        """
        pass
