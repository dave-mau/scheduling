from .clock import Clock, TimeMs
from itertools import chain


class Simulator:
    MAX_NUM_ORDERINGS = 10

    def __init__(self, clock: Clock, dt: TimeMs):
        self._clock = clock
        self._dt = dt
        self._nodes = [[] for _ in range(self.MAX_NUM_ORDERINGS)]

    @property
    def time(self) -> TimeMs:
        return self._clock.get_time_ms()

    def add_node(self, node, order: int):
        assert (
            order < self.MAX_NUM_ORDERINGS
        ), f"Simulator does not support more than {self.MAX_NUM_ORDERINGS} orderings ({order} was passed)."
        self._nodes[order].append(node)

    def run_until(self, t_end: TimeMs):
        while self._clock.get_time_ms() < t_end:
            self.advance()

    def init(self):
        self._update()

    def advance(self):
        self._clock.advance(self._dt)
        self._update()

    def _update(self):
        for order in self._nodes:
            for node in order:
                node.update()

    def reset(self):
        self._clock.reset()
        for node in chain(*self._nodes):
            node.reset()
