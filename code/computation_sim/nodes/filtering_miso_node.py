from copy import deepcopy
from typing import Callable, Generator, List, Optional

import numpy as np
from computation_sim.basic_types import Header, Message, NodeId, Time
from computation_sim.time import DurationSampler, TimeProvider, as_age

from .interfaces import Node, StateVariableNormalizer
from .state_normalizers import ConstantNormalizer


class FilteringMISONode(Node):
    def __init__(
        self,
        time_provider: TimeProvider,
        duration_sampler: DurationSampler,
        id: NodeId = None,
        filter_threshold=np.inf,
        age_normalizer: StateVariableNormalizer = None,
        occupancy_normalizer: StateVariableNormalizer = None,
        count_normalizer: StateVariableNormalizer = None,
        receive_cb: Callable[["FilteringMISONode"], None] = None,
    ):
        super().__init__(time_provider, id)
        self._duration_sampler = duration_sampler
        self.filter_threshold = filter_threshold
        self._input_messages = []
        self._output_pass: Node = None
        self._output_fail: Node = None
        self._age_normalizer = age_normalizer if age_normalizer else ConstantNormalizer(1.0)
        self._occupancy_normalizer = occupancy_normalizer if occupancy_normalizer else ConstantNormalizer(1.0)
        self._count_normalizer = count_normalizer if count_normalizer else ConstantNormalizer(1.0)
        self._receive_cb = receive_cb
        self.reset()

    @property
    def outputs(self) -> List[Node]:
        return [self._output_pass, self._output_fail]

    def receive(self, message: Message) -> None:
        self._input_messages.append(deepcopy(message))
        if self._receive_cb:
            self._receive_cb(self)

    def generate_state(self) -> Generator[float, None, None]:
        yield self._occupancy_normalizer.normalize(float(self.is_busy))
        yield self._age_normalizer.normalize(float(as_age(self._t_start, self.time)))
        yield self._count_normalizer.normalize(float(self._input_count))
        yield self._count_normalizer.normalize(float(self._total_measurement_count))

    @property
    def draw_options(self) -> dict:
        return dict(
            color="darkred" if self.is_busy else "darkgreen",
            symbol="square",
            hovertext=f"is_busy = {self._is_busy}<br>t_start_age = {self.state[1]}",
        )

    def update(self):
        if not self.is_busy:
            # Not busy; nothing to do
            return

        if self.time < self._t_stop:
            # Not yet finished; nothing to do
            return

        # If we reach this, a task has ended. Take necessary action.
        if self._output_pass and (self._result is not None):
            # Has output + result -> send to pass output
            self._output_pass.receive(self._result)
        self._is_busy = False

    def trigger(self):
        if self.is_busy:
            return

        filt_inputs = self._filter_inputs(self._input_messages)
        self._update_input_filter(filt_inputs)
        self._result = self._compute_result(filt_inputs)
        if self._result:
            # Start task iff there is actually a result
            self._set_task_timer()
        self._input_messages.clear()

    def reset(self):
        self._input_messages.clear()
        self._is_busy = False
        self._t_start: Time = self.time
        self._duration: Time = self.time
        self._t_stop: Time = self.time
        self._result = None
        self._input_count = 0
        self._total_measurement_count = 0

    @property
    def is_busy(self) -> bool:
        return self._is_busy

    def set_output_pass(self, node: Node):
        self._output_pass = node

    def set_output_fail(self, node: Node):
        self._output_fail = node

    def _filter_inputs(self, inputs: List[Message]) -> List[Message]:
        # Guard against None or empty list
        valid_inputs: List[Message] = list(filter(lambda x: x is not None, inputs))
        if len(valid_inputs) == 0:
            return []

        # Filter elements that are too old
        youngest_time = max(i.header.t_measure_youngest for i in valid_inputs)

        accepted_inputs = []
        for input in valid_inputs:
            if (self._output_fail is not None) and (
                input.header.t_measure_oldest + self.filter_threshold < youngest_time
            ):
                self._output_fail.receive(input)
            else:
                accepted_inputs.append(input)
        return accepted_inputs

    def _compute_result(self, inputs: List[Message]) -> Optional[Message]:
        if len(inputs) == 0:
            return None

        result = Message(Header())
        result.header.t_measure_oldest = min(i.header.t_measure_oldest for i in inputs)
        result.header.t_measure_youngest = max(i.header.t_measure_youngest for i in inputs)
        result.header.num_measurements = sum(i.header.num_measurements for i in inputs)
        weighted_sum = sum(i.header.num_measurements * i.header.t_measure_average for i in inputs)
        result.header.t_measure_average = round(weighted_sum / result.header.num_measurements)
        return result

    def _set_task_timer(self):
        self._t_start = self.time
        self._duration = self._duration_sampler.sample()
        self._t_stop = self._t_start + self._duration
        self._is_busy = True

    def _update_input_filter(self, inputs: List[Message]):
        self._input_count = len(inputs)
        self._total_measurement_count = sum(i.header.num_measurements for i in inputs)

    def set_receive_cb(self, callback: Callable[["FilteringMISONode"], None]) -> None:
        self._receive_cb = callback
