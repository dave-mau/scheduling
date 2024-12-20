from typing import Iterable, List

import networkx as nx
from computation_sim.basic_types import BadActionError, BadNodeGraphError
from computation_sim.nodes import Node

from .action import Action


class System:
    def __init__(self):
        self._update_list: List[Node] = []
        self._update_list_set = False
        self._actions: List[Action] = []
        self._node_graph: nx.DiGraph = nx.DiGraph()

    @property
    def num_nodes(self) -> int:
        return self._node_graph.number_of_nodes()

    @property
    def num_action(self) -> int:
        return len(self._actions)

    @property
    def actions(self) -> List[Action]:
        return self._actions

    @property
    def state(self) -> List[float]:
        state = []
        for node in self._update_list:
            state.extend(node.generate_state())
        return state

    @property
    def node_graph(self) -> nx.DiGraph:
        return self._node_graph.copy(as_view=True)

    def add_node(self, node: Node) -> None:
        self._node_graph.add_node(node)
        for output in node.outputs:
            self._node_graph.add_edge(node, output)
        self._update_list_set = False

    def add_action(self, action: Action) -> None:
        self._actions.append(action)

    def update(self):
        if not self._update_list_set:
            self._compute_update_list()

        for node in self._update_list:
            node.update()

    def act(self, actions: Iterable[int]):
        if len(actions) != len(self._actions):
            raise BadActionError(
                f"The action has {len(actions)} elements, but the system has {len(self._actions)} actions defined."
            )

        for is_high, action in zip(actions, self._actions):
            if is_high:
                action.act()

    def _compute_update_list(self) -> None:
        self._check_cycles()
        self._check_weakly_connected()
        self._update_list = list(nx.lexicographical_topological_sort(self._node_graph, key=lambda x: x.id))
        self._update_list_set = True

    def _check_cycles(self) -> None:
        if any(nx.simple_cycles(self._node_graph)):
            raise BadNodeGraphError(f"The node graph is invalid, because it has directed cycles.")

    def _check_weakly_connected(self) -> None:
        if not nx.is_weakly_connected(self._node_graph):
            raise BadNodeGraphError(f"The node graph is not weakly connected. Did you forget to set any outputs?")

    def reset(self) -> None:
        for node in self._update_list:
            node.reset()
