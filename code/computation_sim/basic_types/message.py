from typing import Dict, Any
from .types import NodeId
from collections import namedtuple

Header = namedtuple(
    "Header",
    (
        "sender_id",
        "destination_id",
        "oldest_measurement",
        "newest_measurement",
        "average_measurement",
    ),
)


class Message(object):
    def __init__(self, header: Header, data: object):
        self.header = header
        self.data = data
