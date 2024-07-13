import uuid
from collections import namedtuple
from copy import deepcopy
from typing import Callable

from .message_loss_counter import MessageLossCounter

Message = namedtuple("Message", ["min_time", "avg_time", "max_time"])


class Buffer(object):
    """Intermediate storage class that can store exactly one message.

    Messages must be passed in between compute nodes. This class is a simple
    buffer that can store exactly one message.
    """

    def __init__(
        self,
        id: str = None,
        enter_write_cb: Callable[[Message], None] = None,
        exit_write_cb: Callable[[Message], None] = None,
        message_loss_counter: MessageLossCounter = None,
    ):
        self._element = None
        self._id = str(uuid.uuid4()) if id is None else id
        self._enter_write_cb = enter_write_cb
        self._exit_write_cb = exit_write_cb
        self._message_loss_counter = message_loss_counter

    @property
    def id(self):
        """Unique id associated with this buffer."""
        return self._id

    @property
    def has_element(self) -> bool:
        """Check if this instance is currently holding an entry."""
        return self._element is not None

    def set_message_loss_counter(self, message_loss_ocunter: MessageLossCounter):
        self._message_loss_counter = message_loss_ocunter

    def enter_write_cb(self, message: Message):
        if self._enter_write_cb:
            self._enter_write_cb(message)

    def exit_write_cb(self, message: Message):
        if self._exit_write_cb:
            self._exit_write_cb(message)

    def register_enter_write_cb(self, cb: Callable[[Message], None]):
        """Register a callback function that will be executed upon entering the write function."""
        self._enter_write_cb = cb

    def register_exit_write_cb(self, cb: Callable[[Message], None]):
        """Register a callback function that will be executed upon exiting the write function."""
        self._exit_write_cb = cb

    def write(self, element: Message):
        self.enter_write_cb(self._element)
        if self._element and self._message_loss_counter:
            self._message_loss_counter.register_loss(self.id, 1)
        self._element = deepcopy(element)
        self.exit_write_cb(self._element)

    def read(self) -> Message:
        return deepcopy(self._element) if self._element else None

    def clear(self):
        """Sets the buffer contents to None."""
        self._element = None

    def reset(self):
        self.clear()
