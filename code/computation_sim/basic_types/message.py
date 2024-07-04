from typing import Dict, Any
from .types import NodeId
from collections import namedtuple

Header = namedtuple(
    "Header",
    (
        "sender_id",
        "destination_id",
        "t_measure_oldest",
        "t_measure_youngest",
        "t_measure_average",
    ),
)


class Message(object):
    def __init__(self, header: Header, data: object = None):
        self.header = header
        self.data = data
