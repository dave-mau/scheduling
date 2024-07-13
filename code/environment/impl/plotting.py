import tempfile
from itertools import chain
from pathlib import Path

import numpy as np

from .clock import TimeMs
from .system import StateVector, System


def draw_system(system: System, **kwargs) -> object:
    G = system.to_plot_graph()
    return G.pipe(**kwargs)


class SystemStatePlot:
    def __init__(self, system: System):
        self._system = system
        self.reset()

    def reset(self):
        self.times = []
        self.states = []
        self.actions = []

    def names(self):
        names = []
        for _, node in enumerate(self._system.state_nodes):
            names.append(f"{node.id}")
        for _, node in enumerate(self._system.state_nodes):
            names.append(f"{node.output.id}")
        for node in self._system.action_nodes:
            names.append(f"{node.id}_ACTION")
        return names

    def update(self, time: TimeMs, action, state):
        self.times.append(time)
        self.states.append([*state.compute_running, *state.buf_out_has_value, *action])

    def _findones(self, a):
        isone = np.concatenate(([0], a, [0]))
        absdiff = np.abs(np.diff(isone))
        ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
        return np.clip(ranges, 0, len(a) - 1)

    def plot(self, ax):
        data = np.array(self.states)
        names = self.names()
        times = np.array(self.times)

        for channel_idx, _ in enumerate(names):
            indexes = self._findones(data[:, channel_idx])
            for idx in indexes:
                if idx[0] == idx[1]:
                    idx[1] = idx[1] + 1
                ax.hlines(
                    y=channel_idx,
                    xmin=times[idx[0]],
                    xmax=times[idx[1]],
                    linewidth=5,
                    colors="r",
                )

        ax.set_yticks(np.arange(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel("Time [ms]")
        ax.set_xticks(np.arange(times[0], times[-1], 100))
        ax.grid(True)
