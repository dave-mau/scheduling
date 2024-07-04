from .clock import TimeMs, TimeProvider, round_to_milliseconds
from .messaging import Buffer, Message
from .time_samplers import ExecutionTimeSampler
from .compute_task import ComputeTask, ComputeTaskInfo
from .message_loss_counter import MessageLossCounter
from typing import List, Tuple
from enum import Enum
import uuid


class Node:
    def __init__(
        self,
        id: str = None,
        output: Buffer = None,
    ):
        self._id = str(uuid.uuid4()) if id is None else id
        self._output = Buffer() if output is None else output

    @property
    def id(self) -> str:
        return self._id

    @property
    def output(self) -> Buffer:
        return self._output

    def update(self):
        pass

    def reset(self):
        self._output.reset()


class ProcessingState(Enum):
    STARTUP = 0
    IDLE = 1
    BUSY = 2


class ProcessingNode(Node):
    def __init__(
        self,
        compute_task: ComputeTask,
        id: str = None,
        output: Buffer = None,
        message_loss_counter: MessageLossCounter = None,
    ):
        """A compute node that has a set of inputs and a single output.

        This class represents a compute node that has a set of inputs and a single output.
        Upon calling `ProcessingNode.trigger_compute` this class will simulate a computation
        task. Once the task is completed, its result will be written to the output.

        Args:
            compute_task (ComputeTask): Computation task that is executed by this node.
            destination (Buffer): Write-buffer.
            sources (List[Buffer], optional): Read-buffers. Defaults to [].
        """
        super().__init__(id=id, output=output)
        self._compute_task = compute_task
        self._inputs = []
        self._state = ProcessingState.STARTUP
        self._message_loss_counter = message_loss_counter

    @property
    def state(self) -> ProcessingState:
        """Current processing state. Can be STARTUP, BUSY and IDLE."""
        return self._state

    @property
    def compute_task(self) -> ComputeTask:
        return self._compute_task

    @property
    def inputs(self) -> List[Buffer]:
        return self._inputs

    def set_message_loss_counter(self, message_loss_counter):
        self._message_loss_counter = message_loss_counter

    def add_input(self, input: Buffer):
        """Add an input source."""
        self._inputs.append(input)

    def update(self):
        """Update the compute node simulation to the current time."""
        if self._state == ProcessingState.IDLE:
            self._state = self._update_idle()
        elif self._state == ProcessingState.BUSY:
            self._state = self._update_busy()
        elif self._state == ProcessingState.STARTUP:
            self._state = self._update_startup()

    def reset(self):
        super().reset()
        self._compute_task.reset()
        self._state = ProcessingState.STARTUP

    def trigger_compute(self):
        """Trigger the compute task by calling this method.

        If the node is already busy, nothing will happen. If it is in STARTUP or IDLE state,
        it will copy the inputs and wipe the input buffers. It will then compute a result and stay
        BUSY until the task duration has passed.

        Returns:
            dict: Dictionary containing information about the immediate action response. The field
                "executed" indicates, if the action could be executed. Other fields are copied from
                the processor info
                "was_executed" is always set (True, if the action was triggered. False, if not.).
        """
        if self._state == ProcessingState.BUSY:
            return None
        inputs = self.read_inputs()
        self._compute_task.run(inputs)
        self._state = ProcessingState.BUSY
        if self._message_loss_counter:
            self._message_loss_counter.register_loss(self.id, self._compute_task.compute_info.num_rejected_inputs)

    def read_inputs(self) -> List[Message]:
        """Collects copies of the messages in the input buffers and wipes all input buffers afterwards."""
        inputs = []
        for source in self._inputs:
            inputs.append(source.read())
            source.clear()
        return inputs

    def _update_idle(self) -> ProcessingState:
        return ProcessingState.IDLE

    def _update_busy(self) -> ProcessingState:
        if self._compute_task.is_finished:
            self.output.write(self._compute_task.result)
            return ProcessingState.IDLE
        return ProcessingState.BUSY

    def _update_startup(self) -> ProcessingState:
        return ProcessingState.STARTUP


class InputNode(Node):
    def __init__(
        self,
        time_provider: TimeProvider,
        tauMs: TimeMs,
        t0Ms: TimeMs,
        disturbance_sampler: ExecutionTimeSampler,
        id: str = None,
        output: Buffer = None,
    ):
        """Input compute node that generates outputs at fixed time intervals.

        This class models an input node that does not have any inputs, but only generates outputs.
        Outputs are generated at times `t_i = tauMs * i + t0Ms + epsilon`, where `i` is any integer,
        `tauMs` is the period, `t0Ms` the phase and `epsilon` some random perturbation.

        Args:
            time_provider (TimeProvider): Permits access to read the current time.
            tauMs (TimeMs): Periodicity of the input signal (millisecond).
            t0Ms (TimeMs): Phase offset of the input signal (millisecond).
            disturbance_sampler (ExecutionTimeSampler): Distribution used to sample epsilon.
            id (str, optional): Unique ID of this instance. Defaults to str(uuid.uuid4()).
            output (Buffer, optional): Destination where outputs are written to. Defaults to Buffer().
        """
        super().__init__(id, output)
        self._time_provider = time_provider
        self._tauMs = tauMs
        self._t0Ms = t0Ms
        self._disturbance_sampler = disturbance_sampler

        self._t_trigger_nominal = self.nominal_next_trigger_time()
        self._t_trigger_actual = self.next_trigger_time()

    def update(self):
        curr_time = self._time_provider.time
        if curr_time >= self._t_trigger_actual:
            self.output.write(Message(curr_time, curr_time, curr_time))
            self._t_trigger_nominal += self._tauMs
            self._t_trigger_actual = self.next_trigger_time()

    def reset(self):
        super().reset()
        self._t_trigger_nominal = self.nominal_next_trigger_time()
        self._t_trigger_actual = self.next_trigger_time()

    @round_to_milliseconds
    def next_trigger_time(self):
        return self._t_trigger_nominal + self._disturbance_sampler.sample()

    @round_to_milliseconds
    def nominal_last_trigger_time(self):
        return self._time_provider.time - (self._time_provider.time - self._t0Ms) % self._tauMs

    @round_to_milliseconds
    def nominal_next_trigger_time(self):
        return self.nominal_last_trigger_time() + self._tauMs
