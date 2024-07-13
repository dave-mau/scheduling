from abc import ABC
from typing import List

import graphviz
import networkx as nx
import numpy as np

from .clock import TimeMs, TimeProvider
from .message_loss_counter import MessageLossCounter
from .nodes import InputNode, ProcessingNode
from .simulator import Simulator


class StateVector:
    NUM_STATES_PER_NODE = 6

    def __init__(self, num_nodes: int, data: np.ndarray = None):
        expected_shape = (StateVector.NUM_STATES_PER_NODE * num_nodes,)
        if data:
            assert (
                data.shape == expected_shape
            ), f"The shape of the state vector you passed is wrong (shape is {data.shape}, but should be {expected_shape})."
            self.data = data
        else:
            self.data = np.zeros(expected_shape, dtype=float)
        self._num_nodes = num_nodes

    @property
    def compute_start_ages(self) -> np.ndarray:
        return self.data[0 : self._num_nodes]

    @property
    def buf_out_min_ages(self) -> np.ndarray:
        return self.data[self._num_nodes : 2 * self._num_nodes]

    @property
    def buf_out_avg_ages(self) -> np.ndarray:
        return self.data[2 * self._num_nodes : 3 * self._num_nodes]

    @property
    def buf_out_max_ages(self) -> np.ndarray:
        return self.data[3 * self._num_nodes : 4 * self._num_nodes]

    @property
    def compute_running(self) -> np.ndarray:
        return self.data[4 * self._num_nodes : 5 * self._num_nodes]

    @property
    def buf_out_has_value(self) -> np.ndarray:
        return self.data[5 * self._num_nodes : 6 * self._num_nodes]


class BadSystemArchitectureError(ValueError):
    pass


class DuplicateNodeError(ValueError):
    def __init__(self, id: str):
        super().__init__(f"The node with name {id} already exists in the system graph. Cannot add it twice.")


class ConnectionNextNodeMissingError(ValueError):
    def __init__(self, prev: str, next: str):
        super().__init__(f"Cannot create a connection from {prev} to {next}, because {next} does not exist.")


class System(object):
    def __init__(self, time_provider: TimeProvider, message_loss_counter: MessageLossCounter = None):
        self._time_provider = time_provider
        self._input_nodes: List[InputNode] = []
        self._action_nodes: List[ProcessingNode] = []
        self._state_nodes: List[ProcessingNode] = []
        self.message_loss_counter = message_loss_counter if message_loss_counter else MessageLossCounter()
        self._graph = nx.DiGraph()

    @property
    def num_state_nodes(self) -> int:
        return len(self._state_nodes)

    @property
    def num_action_nodes(self) -> int:
        return len(self._action_nodes)

    @property
    def num_states(self) -> int:
        return StateVector.NUM_STATES_PER_NODE * self.num_state_nodes

    @property
    def input_nodes(self) -> List[InputNode]:
        return self._input_nodes.copy()

    @property
    def action_nodes(self) -> List[ProcessingNode]:
        return self._action_nodes.copy()

    @property
    def state_nodes(self) -> List[ProcessingNode]:
        return self._state_nodes.copy()

    def reset(self):
        for node in self._input_nodes:
            node.reset()
        for node in self._state_nodes:
            node.reset()
        if self.message_loss_counter:
            self.message_loss_counter.reset()

    def update(self):
        for node in self._input_nodes:
            node.update()
        for node in self._state_nodes:
            node.update()

    def get_state(self, age_normalization_factor: TimeMs = 100) -> StateVector:
        to_normed_age = lambda t: float(self._time_provider.time - t) / float(age_normalization_factor)

        state = StateVector(self.num_state_nodes)
        for idx, node in enumerate(self._state_nodes):
            state.compute_start_ages[idx] = to_normed_age(node.compute_task.t_start)
            state.compute_running[idx] = node.compute_task.is_running
            state.buf_out_has_value[idx] = node.output.has_element
            if state.buf_out_has_value[idx]:
                msg = node.output.read()
                state.buf_out_min_ages[idx] = to_normed_age(msg.min_time)
                state.buf_out_avg_ages[idx] = to_normed_age(msg.avg_time)
                state.buf_out_max_ages[idx] = to_normed_age(msg.max_time)
        return state

    def act(self, action: np.ndarray):
        for action_idx, node in enumerate(self._action_nodes):
            if action[action_idx] == 1:
                node.trigger_compute()

    def add_processing_node(self, node: ProcessingNode, next_node_id: str, is_state_node=True, is_action_node=True):
        if node.id in self._graph.nodes:
            raise DuplicateNodeError(node.id)
        if next_node_id and (not next_node_id in self._graph.nodes):
            raise ConnectionNextNodeMissingError(node.id, next_node_id)

        self._graph.add_node(
            node.id, proc_node_object=node, is_state_node=is_state_node, is_action_node=is_action_node
        )
        if next_node_id:
            self._graph.add_edge(node.id, next_node_id)
            self._get_proc_node(next_node_id).add_input(node.output)

    def add_input_node(self, node: InputNode, next_node_id: str, trigger_next_compute=False):
        if node.id in self._graph.nodes:
            raise DuplicateNodeError(node.id)
        if next_node_id not in self._graph.nodes:
            raise ConnectionNextNodeMissingError(node.id, next_node_id)

        self._graph.add_node(node.id, input_node_object=node, is_state_node=False, is_action_node=False)
        self._graph.add_edge(node.id, next_node_id)
        next_node = self._get_proc_node(next_node_id)
        next_node.add_input(node.output)
        if trigger_next_compute:
            node.output.register_exit_write_cb(lambda _: next_node.trigger_compute())

    def compile(self):
        if nx.function.is_empty(self._graph):
            return
        if not nx.is_tree(self._graph):
            raise BadSystemArchitectureError(f"The system you specified does not form a tree!")
        self._input_nodes = self._collect_input_nodes()
        self._state_nodes = self._collect_state_nodes()
        self._action_nodes = self._collect_action_nodes()

        # Register message loss counter
        for node in self._state_nodes:
            node.set_message_loss_counter(self.message_loss_counter)
            node.output.set_message_loss_counter(self.message_loss_counter)
        for node in self._input_nodes:
            node.output.set_message_loss_counter(self.message_loss_counter)
        self._state_nodes[-1].output.set_message_loss_counter(None)

    def _collect_input_nodes(self) -> List[InputNode]:
        # Find all nodes that dont have any in-pointing edges.
        indices, _ = zip(*filter(lambda element: element[1] == 0, self._graph.in_degree()))
        return [self._graph.nodes[id]["input_node_object"] for id in indices]

    def _collect_state_nodes(self) -> List[ProcessingNode]:
        sorted_node_ids = list(nx.lexicographical_topological_sort(self._graph))
        filt_node_ids = filter(lambda id: self._graph.nodes[id].get("is_state_node"), sorted_node_ids)
        return [self._graph.nodes[id]["proc_node_object"] for id in filt_node_ids]

    def _collect_action_nodes(self) -> List[ProcessingNode]:
        sorted_node_ids = list(nx.lexicographical_topological_sort(self._graph))
        filt_node_ids = filter(lambda id: self._graph.nodes[id].get("is_action_node"), sorted_node_ids)
        return [self._graph.nodes[id]["proc_node_object"] for id in filt_node_ids]

    def _get_proc_node(self, id: str) -> ProcessingNode:
        return self._graph.nodes(data=True)[id].get("proc_node_object")

    def add_to_sim(self, sim: Simulator):
        for node in self._input_nodes:
            sim.add_node(node, 0)
        for node in self._state_nodes:
            sim.add_node(node, 1)

    def to_plot_graph(self) -> graphviz.Digraph:
        G = graphviz.Digraph()
        for node in self._input_nodes:
            G.node(node.id, label=node.id, shape="cylinder", fillcolor="white", style="filled")
            G.node(
                node.output.id,
                label=node.output.id,
                shape="ellipse",
                fillcolor="red" if node.output.has_element else "green",
                style="filled",
            )
            G.edge(node.id, node.output.id)
            for edge in self._graph.out_edges([node.id]):
                G.edge(node.output.id, edge[1])
        for node in self._state_nodes:
            G.node(
                node.id,
                label=node.id,
                shape="square",
                color="magenta" if self._graph.nodes(data=True)[node.id].get("is_action_node") else "black",
                penwidth="4.0",
                fillcolor="red" if node.compute_task.is_running else "green",
                style="filled",
            )
            G.node(
                node.output.id,
                label=node.output.id,
                shape="ellipse",
                fillcolor="red" if node.output.has_element else "green",
                style="filled",
            )
            G.edge(node.id, node.output.id)
            for edge in self._graph.out_edges([node.id]):
                G.edge(node.output.id, edge[1])
        return G


class SystemBuilder(ABC):
    def __init__(self, time_provider: TimeProvider):
        self._time_provider = time_provider
        self._system = System(self._time_provider)

    @property
    def system(self) -> System:
        return self._system

    def reset(self, time_provider: TimeProvider):
        self._time_provider = time_provider
        self._system = System(self._time_provider)
