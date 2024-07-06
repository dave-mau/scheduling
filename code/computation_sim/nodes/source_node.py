from computation_sim.basic_types import (
    NodeId,
    Message,
    CommunicationError,
    Time,
    Header,
)
from computation_sim.time import DurationSampler
from .node import Node
from abc import ABC, abstractmethod
from typing import List, Optional
from copy import deepcopy


class Sensor(ABC):
    def __init__(self):
        self._last_update_time: Optional[Time] = None
        self._has_measurement = False

    @property
    @abstractmethod
    def state(self) -> List[float]:
        pass

    @property
    def has_measurement(self) -> bool:
        return self._has_measurement

    @abstractmethod
    def get_measurement(self) -> Optional[Message]:
        pass

    def update(self, time: Time):
        self._last_update_time = time

    def reset(self) -> None:
        self._last_update_time = None
        self._has_measurement = False


class PeriodicEpochSensor(Sensor):
    def __init__(
        self, epoch: Time, period: Time, disturbance: DurationSampler, **kwargs
    ):
        super().__init__(**kwargs)
        self._epoch = epoch
        self._period = period
        self._disturbance = disturbance
        self._nominal_send_time = self._epoch
        self._actual_send_time = self._nominal_send_time

    @property
    def state(self) -> List[float]:
        return []

    def get_measurement(self) -> Message | None:
        if self.has_measurement:
            result = Message(self._get_header())
            result.data = self._get_data()
            return result
        return None

    def update(self, time: Time):
        super().update(time)
        if time >= self._actual_send_time:
            self._nominal_send_time += self._period
            while self._actual_send_time <= time:
                self._actual_send_time = (
                    self._nominal_send_time + self._disturbance.sample()
                )
            self._has_measurement = True
            return
        self._has_measurement = False

    def reset(self):
        super().reset()
        self._nominal_send_time = self._epoch
        self._actual_send_time = self._nominal_send_time

    def _get_header(self) -> Header:
        header = Header(
            self._last_update_time, self._last_update_time, self._last_update_time
        )
        return header

    def _get_data(self) -> object:
        return {}


class SourceNode(Node):
    def __init__(self, sensor: Sensor, id: NodeId = None):
        super().__init__(id)
        self._sensor = sensor

    def receive(self, message: Message) -> None:
        raise CommunicationError("Source node cannot receive a message.")

    @property
    def state(self) -> List[float]:
        return self._sensor.state

    def update(self, time: Time):
        self._sensor.update(time)
        message = self._sensor.get_measurement()
        if message:
            self._send(message)

    def trigger(self):
        pass

    def reset(self):
        self._sensor.reset()

    def add_output(self, output: Node) -> None:
        if output in self.outputs:
            raise ValueError(
                f"The node with id {output.id} cannot be added twice as output."
            )
        self._outputs.append(output)

    def _send(self, message: Message):
        for output in self._outputs:
            message = deepcopy(message)
            message.header.sender_id = self.id
            message.header.destination_id = output.id
            output.receive(deepcopy(message))
