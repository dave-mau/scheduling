from itertools import chain
from typing import Dict, List, Generator

from computation_sim.basic_types import (
    BadNodeGraphError,
    CommunicationError,
    Message,
    NodeId,
)
from computation_sim.time import TimeProvider

from .interfaces import Node, NodeVisitor


class BufferedComputeNode(Node):
    def __init__(
        self,
        time_provider: TimeProvider,
        compute: Node,
        id: NodeId = None,
    ):
        super().__init__(time_provider=time_provider, id=id)
        self._input_buffers: Dict[NodeId, Node] = dict()
        self._compute = compute

    @property
    def outputs(self) -> List["Node"]:
        return self._compute.outputs

    @property
    def compute_node(self) -> Node:
        return self._compute

    def set_buffer_for_sender(self, input_id: NodeId, buffer: Node):
        if self._compute not in buffer.outputs:
            # The node interface does not allow us to add an output to Nodes.
            # We could specify a buffer interface, but this would impose constraints on
            # how outputs can be added to a buffer. We thus leave responsibility for
            # adding the compute node as buffer output to users and raise an error
            # to make users aware of this possible pitfall.
            raise BadNodeGraphError(
                f"Cannot add buffer {buffer.id}, because the compute node ({self._compute.id}) is not one of its outputs."
            )
        self._input_buffers[input_id] = buffer

    def receive(self, message: Message) -> None:
        if not message.header.sender_id in self._input_buffers.keys():
            raise CommunicationError(
                f"BufferedComputeNode with id {id} received a message from sender {message.header.sender_id}, but no input buffer is set for this sender."
            )
        self._input_buffers[message.header.sender_id].receive(message)

    def generate_state(self) -> Generator[float, None, None]:
        for buffer in self._input_buffers.values():
            yield from buffer.generate_state()
        yield from self._compute.generate_state()

    def update(self):
        for buf in self._input_buffers.values():
            buf.update()
        self._compute.update()

    def trigger(self):
        for buf in self._input_buffers.values():
            buf.trigger()
        self._compute.trigger()

    def visit(self, visitor: NodeVisitor):
        for buf in self._input_buffers.values():
            buf.visit(visitor)
        self._compute.visit(visitor)

    def reset(self):
        for buf in self._input_buffers.values():
            buf.reset()
        self._compute.reset()
