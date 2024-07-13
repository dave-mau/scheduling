from copy import deepcopy
from typing import List

from computation_sim.basic_types import CommunicationError, Message, NodeId
from computation_sim.time import TimeProvider

from .interfaces import Node, Sensor


class SourceNode(Node):
    def __init__(self, time_provider: TimeProvider, sensor: Sensor, id: NodeId = None):
        super().__init__(time_provider, id)
        self._sensor = sensor

    def receive(self, message: Message) -> None:
        raise CommunicationError("Source node cannot receive a message.")

    @property
    def state(self) -> List[float]:
        return self._sensor.state

    def update(self):
        self._sensor.update(self.time)
        message = self._sensor.get_measurement()
        if message:
            self._send(message)

    def trigger(self):
        pass

    def reset(self):
        self._sensor.reset()

    def add_output(self, output: Node) -> None:
        if output in self.outputs:
            raise ValueError(f"The node with id {output.id} cannot be added twice as output.")
        self._outputs.append(output)

    def _send(self, message: Message):
        for output in self._outputs:
            message = deepcopy(message)
            message.header.sender_id = self.id
            message.header.destination_id = output.id
            output.receive(deepcopy(message))
