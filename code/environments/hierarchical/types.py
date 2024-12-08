from typing import List, NamedTuple

from computation_sim.nodes import (
    FilteringMISONode,
    OutputNode,
    RingBufferNode,
    SinkNode,
    SourceNode,
)
from computation_sim.system import Action, System


class ActionCollection(NamedTuple):
    input_buffers: List[RingBufferNode]
    node: FilteringMISONode
    action: Action


class SystemCollection(NamedTuple):
    system: System
    sources: List[SourceNode]
    sinks: List[SinkNode]
    action_collections: List[ActionCollection]
    output: OutputNode
