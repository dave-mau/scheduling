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


class SourceStrategy(ABC):
    @abstractmethod
    def update(self, time: Time) -> Optional[object]:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


class SourceNode(Node):
    def __init__(self, strategy: SourceStrategy, id: NodeId = None):
        super().__init__(id)
        self._strategy = strategy
        self._last_send_time: Optional[Time] = None

    def receive(self, message: Message) -> None:
        super().receive(message)
        raise CommunicationError("Source node cannot receive a message.")

    def add_output(self, output: Node) -> None:
        super().add_output(output)
        self._outputs[output.id] = output

    @property
    def state(self) -> List[float]:
        return [self._last_send_time]

    def update(self, time: Time):
        measurement = self._strategy.update(time)
        if measurement:
            for output in self._outputs.values():
                header = Header(self.id, output.id, time, time, time)
                output.receive(Message(header, measurement))
                self._last_send_time = time

    def reset(self):
        self._strategy.reset()
        self._last_send_time = None


class PeriodicEpochSender(SourceStrategy):
    def __init__(self, epoch: Time, period: Time, disturbance: DurationSampler):
        self.epoch = epoch
        self.period = period
        self.disturbance = disturbance
        self.nominal_send_time = self.epoch
        self.actual_send_time = self.nominal_send_time

    def update(self, time: Time) -> Optional[object]:
        if time >= self.actual_send_time:
            self.nominal_send_time += self.period
            while self.actual_send_time <= time:
                self.actual_send_time = (
                    self.nominal_send_time + self.disturbance.sample()
                )
            return {}
        return None

    def reset(self):
        self.nominal_send_time = self.epoch
        self.actual_send_time = self.nominal_send_time
