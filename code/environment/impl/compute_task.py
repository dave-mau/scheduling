from .clock import TimeProvider, TimeMs
from .time_samplers import ExecutionTimeSampler
from typing import List
from .messaging import Message
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Tuple
import numpy as np

ComputeTaskInfo = namedtuple("ComputeTaskInfo", ["num_inputs", "num_rejected_inputs"])


class ComputeTask(ABC):

    def __init__(self, time_provider: TimeProvider, execution_time: ExecutionTimeSampler):
        """Compute tasks that accepts a set of symbolic inputs and returns a single, symbolic output

        The compute task takes a duration that is sampled from the `execution_time` samples.
        Only after completion, the output is accessible. Thus, a compute task has a starting time,
        a duration and a stopping time.

        Args:
            time_provider (TimeProvider): Time source.
            execution_time (ExecutionTimeSampler): Sampler that can be used to sample the task duration.
        """
        self._time_provider = time_provider
        self._execution_time = execution_time

        self._t_start = time_provider.time
        self._t_end = time_provider.time
        self._duration = time_provider.time
        self._result = None
        self._compute_info = None

    @property
    def is_finished(self) -> bool:
        """Check if the task is finished."""
        return self._time_provider.time >= self._t_end

    @property
    def is_running(self) -> bool:
        """Check if the task is running."""
        return not self.is_finished

    @property
    def t_start(self) -> TimeMs:
        """Starting time of the current task. Time is zero, if the task did not start yet."""
        return self._t_start

    @property
    def duration(self) -> TimeMs:
        """Returns the duration taken by the last, finished task."""
        return self._duration if self.is_finished else None

    @property
    def t_end(self) -> TimeMs:
        """Returns the end time of the currently active task."""
        return self._t_end if self.is_finished else None

    @property
    def result(self) -> Message:
        """Result of the current task. None, if the task did not start yet."""
        return self._result if self.is_finished else None

    @property
    def compute_info(self) -> ComputeTaskInfo:
        """Information about the last compute operation that is not contained within the result."""
        return self._compute_info

    def run(self, input_args: List[Message]):
        """Given a list of inputs, compute the result and sample the simulated task duration."""
        if self.is_running:
            raise RuntimeError("Cannot run compute task while a task is already running.")
        self._result, self._compute_info, self._duration = self._exec(input_args)

        self._t_start = self._time_provider.time
        self._t_end = TimeMs(self._t_start + self._duration)

    def reset(self):
        self._t_start = self._time_provider.time
        self._t_end = self._time_provider.time
        self._duration = self._time_provider.time
        self._result = None
        self._compute_info = None

    @abstractmethod
    def _exec(self, input_args: List[Message]) -> Tuple[Message, ComputeTaskInfo, TimeMs]:
        pass


class SISOComputeTask(ComputeTask):
    def _exec(self, input_args: List[Message]) -> Tuple[Message, ComputeTaskInfo, TimeMs]:
        if len(input_args) == 1 and input_args[0]:
            return input_args[0], ComputeTaskInfo(1, 0), self._execution_time.sample()
        elif len(input_args) == 1 and not input_args[0]:
            return input_args[0], ComputeTaskInfo(1, 1), 0
        else:
            raise RuntimeError(f"SISOComputeTask must receive exactly one input (got {len(input_args)}).")


class MISOFusionTask(ComputeTask):
    def __init__(
        self,
        time_provider: TimeProvider,
        execution_time: ExecutionTimeSampler,
        filter_thresh: TimeMs,
    ):
        super().__init__(time_provider, execution_time)
        self._filter_thresh = filter_thresh

    def _exec(self, input_args: List[Message]) -> Tuple[Message, ComputeTaskInfo, TimeMs]:
        n_inputs = len(input_args)

        if len(input_args) == 0:
            raise RuntimeError("MISOFusionTask requires at least one input.")

        # Guard: Make sure at least one input is not None
        if self._is_all_none(input_args):
            return None, ComputeTaskInfo(n_inputs, n_inputs), 0

        # Get the time of the newest measurement in the inputs
        max_time = self._get_max_time(input_args)

        # Reject the inputs according to the filtering criterion
        accepted = list(filter(lambda msg: self._filter_criterion(msg, max_time), input_args))

        # Avoid division by zero
        if len(accepted) == 0:
            return None, ComputeTaskInfo(n_inputs, n_inputs), 0

        # Happy path: At least one input accepted
        avg_accepted_time = sum(getattr(msg, "avg_time") for msg in accepted) / len(accepted)
        max_accepted_time = max(getattr(msg, "max_time") for msg in accepted)
        min_accepted_time = min(getattr(msg, "min_time") for msg in accepted)
        return (
            Message(min_accepted_time, avg_accepted_time, max_accepted_time),
            ComputeTaskInfo(n_inputs, n_inputs - len(accepted)),
            self._execution_time.sample(),
        )

    def _is_all_none(self, input_args: List[Message]) -> bool:
        return input_args.count(None) == len(input_args)

    def _get_max_time(self, input_args: List[Message]) -> TimeMs:
        max_times = []
        for input in input_args:
            if input:
                max_times.append(input.max_time)
        return TimeMs(np.max(np.array(max_times)))

    def _filter_criterion(self, msg: Message, max_time: TimeMs) -> bool:
        return (msg is not None) and (msg.min_time + self._filter_thresh) >= max_time
