from computation_sim.nodes import (
    RingBufferNode, FilteringMISONode,SourceNode, SinkNode, OutputNode
)
from computation_sim.system import System, Action
from typing import List
from typing import NamedTuple

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
