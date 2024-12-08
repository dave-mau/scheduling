from collections import defaultdict
from threading import Thread
from typing import Dict, List

import computation_sim.system as system
import gymnasium as gym
import networkx as nx
import numpy as np
from computation_sim.basic_types import Header, Time
from computation_sim.nodes import Node
from computation_sim.system import ImageCreator, SystemDrawer
from computation_sim.time import Clock, as_age
from dash import Dash, Input, Output, dcc, html

from .reward import Reward
from .types import ActionCollection, SystemCollection


def count_upstream_sources(system_collection: SystemCollection) -> Dict[Node, int]:
    """For each node in the graph, count the number of upstream sources that are connected to it."""
    node_graph = system_collection.system.node_graph
    upstream_sensor_count = defaultdict(int)
    for source in system_collection.sources:
        for node in nx.descendants(node_graph, source):
            upstream_sensor_count[node] += 1
    return upstream_sensor_count


def build_system_drawer(env, width=800, height=800) -> SystemDrawer:
    drawer = SystemDrawer()
    drawer.build(env.system.node_graph)
    drawer.fw.update_layout(width=width, height=height)
    return drawer


class InformationLossObserver:
    def __init__(self, system_collection: SystemCollection):
        self._system_collection = system_collection
        self._baseline = self._record_reward_baseline(self._system_collection.action_collections)
        self._sensor_count = count_upstream_sources(system_collection)

    def _record_reward_baseline(self, ac: List[ActionCollection]):
        return list(filter(lambda x: not x.node.is_busy, ac))

    def _get_activated_actions(self):
        return filter(lambda x: x.node.is_busy, self._baseline)

    @property
    def buffer_overrides(self) -> Dict[str, int]:
        """Counts the number of messages that were lost due to buffer overrides."""
        return {sink.id: sink.count for sink in self._system_collection.sinks}

    @property
    def missing_measurements(self) -> Dict[str, int]:
        """For each action node that was activated, count the number of measurements that were not received."""
        counts = dict()
        for collection in self._get_activated_actions():
            counts[collection.node.id] = self._sensor_count[collection.node] - collection.node.total_measurement_count
        return counts

    @property
    def missing_inputs(self) -> Dict[str, int]:
        """Counts the number of missing inputs for each action node that was
        activated since the last time record_basline_state was called.
        """
        # Find all action nodes that were activated since the last time
        # record_basline_state was called.
        counts = dict()
        for collection in self._get_activated_actions():
            counts[collection.node.id] = len(collection.input_buffers) - collection.node.filtered_input_count
        return counts


class HierarchicalSystemBase(gym.Env):
    metadata = {"render.modes": ["human", "image", "jupyter"], "render_fps": 10}

    def __init__(
        self,
        clock: Clock,
        system_collection: SystemCollection,
        reward: Reward,
        dt: Time = 10,
        render_mode=None,
        window_size=(800, 800),
        **kwargs
    ):
        super().__init__(**kwargs)
        # Store init params
        self.clock: Clock = clock
        self._system_collection: SystemCollection = system_collection
        self._reward: Reward = reward
        self._dt = dt

        # Set dimensionality of action / observation spaces
        self.action_space = gym.spaces.Discrete(system.num_actions(self.system.num_action))
        lb = -np.inf * np.ones((len(self.system.state),), dtype=float)
        ub = +np.inf * np.ones((len(self.system.state),), dtype=float)
        self.observation_space = gym.spaces.Box(lb, ub, dtype=float)

        # Setup rendering
        self.render_mode = render_mode
        self.window_size = window_size

        # Drawer converts the system graph into a plotly figure
        self.drawer = None
        if self.render_mode is not None:
            self.drawer = build_system_drawer(self, *self.window_size)

        # Human rendering: Setup dash window
        self.window = None
        if self.render_mode == "human":
            self.window = Dash(__name__)
            self.window.layout = html.Div(
                [dcc.Graph(id="graph"), dcc.Interval(id="interval", interval=1000 / self.metadata["render_fps"])]
            )

            @self.window.callback(Output("graph", "figure"), Input("interval", "n_intervals"))
            def update_graph(n_intervals):
                return self.drawer.fw

            # self.window_port = 8050
            # self.window.run_server(debug=False, port = self.window_port)

        # Human rendering: Dash runs in a separate thread
        self.dash_thread = None

    @property
    def system(self) -> system.System:
        return self._system_collection.system

    @property
    def time(self) -> Time:
        return self.clock.get_time()

    @property
    def state(self) -> np.ndarray:
        return np.array(self.system.state).flatten()

    @property
    def output_age(self) -> Header:
        """Gets the age of the output message."""
        last_received = self._system_collection.output.last_received
        now = self.clock.get_time()
        if last_received:
            return dict(
                output_age_min=(as_age(last_received.header.t_measure_youngest, now)),
                output_age_max=(as_age(last_received.header.t_measure_oldest, now)),
                output_age_avg=(as_age(last_received.header.t_measure_average, now)),
            )
        else:
            return dict(
                output_age_min=(as_age(self.clock.initial_time, now)),
                output_age_max=(as_age(self.clock.initial_time, now)),
                output_age_avg=(as_age(self.clock.initial_time, now)),
            )

    def reset(self, seed=None):
        super().reset(seed=seed)
        for sampler in self._system_collection.samplers:
            sampler.reset(seed=seed)
        self.clock.reset()
        self.system.reset()
        self.system.update()
        return self.state, {}

    def act(self, action: List[int]):
        self.system.act(action)

    def advance(self):
        self.clock += self._dt
        self.system.update()

    def step(self, action: int):
        # Reset the sinks that count number of lost messages
        # This means, we count the number of lost messages from now on.
        for sink in self._system_collection.sinks:
            sink.reset()

        # Set baseline for reward function that
        observer = InformationLossObserver(self._system_collection)

        # Apply the action and advance the system
        action = system.unpack_action(self.system.num_action, action)
        self.act(action)
        self.advance()

        # Construct info
        info = dict(
            buffer_overrides=observer.buffer_overrides,
            missing_inputs=observer.missing_inputs,
            missing_measurements=observer.missing_measurements,
            **self.output_age
        )
        reward = self._reward(action, info["buffer_overrides"], info["missing_measurements"], info["output_age_avg"])

        # Render
        self._draw()
        self.render()

        # Build the reward
        return self.state, reward, False, False, info

    def _draw(self):
        if self.drawer is not None:
            self.drawer.update(self.system.node_graph)

    def render(self):
        if self.render_mode == "image":
            return ImageCreator().create(self.drawer.fw)
        elif self.render_mode == "human":
            if self.dash_thread is None:
                self.dash_thread = Thread(target=lambda: self.window.run_server(debug=False))
                self.dash_thread.daemon = True
                self.dash_thread.start()
            pass
        elif self.render_mode == "jupyter":
            return self.drawer.fw
        else:
            pass
