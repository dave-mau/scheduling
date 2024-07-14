from typing import Iterable, List

from computation_sim.basic_types import BadActionError
from computation_sim.nodes import Node

from .action import Action


class System:
    def __init__(self):
        self._nodes: List[Node] = []
        self._actions: List[Action] = []

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def num_action(self) -> int:
        return len(self._actions)

    def add_node(self, node: Node) -> None:
        self._nodes.append(node)

    def add_action(self, action: Action) -> None:
        self._actions.append(action)

    def compile(self):
        # TODO
        pass

    def update(self):
        for node in self._nodes:
            node.update()

    def act(self, actions: Iterable[int]):
        if len(actions) != len(self._actions):
            raise BadActionError(
                f"The action has {len(actions)} elements, but the system has {len(self._actions)} actions defined."
            )

        for is_high, action in zip(actions, self._actions):
            if is_high:
                action.act()
