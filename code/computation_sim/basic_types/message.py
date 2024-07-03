from typing import List
from .types import Time, NodeId
from typing import Dict, Any


class Message(object):
    def __init__(self, sender_id: NodeId, destination_id: NodeId, data: Dict[str, Any]):
        self._sender_id = sender_id
        self._destination_id = destination_id
        self._data = data

    @property
    def sender_id(self) -> NodeId:
        return self._sender_id

    @property
    def destination_id(self) -> NodeId:
        return self._destination_id

    @property
    def data(self) -> NodeId:
        return self._data
