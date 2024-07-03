from abc import ABC, abstractmethod
from typing import List


class NodeDrawer:
    @property
    def draw_settings(self) -> dict:
        return {
            "shape": "circle",
            "color": "black",
            "penwidth": "1.0",
            "fillcolor": "white",
            "style": "filled",
        }


class Visitor(ABC):
    @abstractmethod
    def visit_node(self, node: "Node"):
        pass


class Node(ABC):
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
    def state(self) -> List["Node"]:
        pass

    @property
    @abstractmethod
    def drawer(self) -> NodeDrawer:
        pass

    @drawer.setter
    @abstractmethod
    def drawer(self, value):
        pass

    def visit(self, visitor: Visitor):
        visitor.visit_node(self)
