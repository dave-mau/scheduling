from unittest.mock import MagicMock

import pytest
from computation_sim.system import Action


def test_priority():
    call_order = []
    action = Action()
    cb1 = MagicMock()
    cb1.side_effect = lambda: call_order.append("cb1")
    cb2 = MagicMock()
    cb2.side_effect = lambda: call_order.append("cb2")

    action.register_callback(cb1, 0, "foo")
    action.register_callback(cb2, 1, "foo")

    action.act()

    assert call_order[0] == "cb2"
    assert call_order[1] == "cb1"


def test_clear():
    action = Action()
    cb1 = MagicMock()
    action.register_callback(cb1, 0, "foo")
    action.act()
    cb1.assert_called_once()

    cb1.reset_mock()
    action.clear()
    action.act()
    cb1.assert_not_called()


def test_len():
    action = Action()
    cb1 = MagicMock()
    cb2 = MagicMock()
    action.register_callback(cb1, 0)
    action.register_callback(cb2, 0)

    assert len(action) == 2


def test_repr():
    action = Action("FooBar")
    action.register_callback(MagicMock(), 0)
    assert str(action) == 'Action "FooBar" with 1 callbacks.'
