from unittest.mock import MagicMock

import numpy as np
import pytest
from computation_sim.system import Action, max_action_id, num_actions, unpack_action


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


def test_max_action_id_0():
    assert max_action_id(0) == 0


def test_max_action_id_1():
    assert max_action_id(1) == 1


def test_max_action_id_2():
    assert max_action_id(2) == 3


def test_max_action_id_3():
    assert max_action_id(3) == 7


def test_num_actions_0():
    assert num_actions(0) == 0


def test_num_actions_1():
    assert num_actions(1) == 2


def test_num_actions_2():
    assert num_actions(2) == 4


def test_num_actions_3():
    assert num_actions(3) == 8


def test_unpack_action_10():
    assert unpack_action(1, 0) == np.array([0])


def test_unpack_action_11():
    assert unpack_action(1, 1) == np.array([1])


def test_unpack_action_30():
    np.testing.assert_equal(unpack_action(3, 0), np.array([0, 0, 0]))


def test_unpack_action_31():
    np.testing.assert_equal(unpack_action(3, 1), np.array([0, 0, 1]))


def test_unpack_action_32():
    np.testing.assert_equal(unpack_action(3, 2), np.array([0, 1, 0]))


def test_unpack_action_33():
    np.testing.assert_equal(unpack_action(3, 3), np.array([0, 1, 1]))


def test_unpack_action_34():
    np.testing.assert_equal(unpack_action(3, 4), np.array([1, 0, 0]))


def test_unpack_action_36():
    np.testing.assert_equal(unpack_action(3, 6), np.array([1, 1, 0]))


def test_unpack_action_37():
    np.testing.assert_equal(unpack_action(3, 7), np.array([1, 1, 1]))
