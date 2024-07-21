from abc import ABC, abstractmethod
from typing import Dict

from computation_sim.nodes import Node

from .system import System


class SystemBuidler(ABC):
    @abstractmethod
    def build(self) -> None:
        pass

    @property
    def system(self) -> System:
        return self._system

    @property
    def nodes(self) -> Dict[str, Node]:
        return self._nodes
