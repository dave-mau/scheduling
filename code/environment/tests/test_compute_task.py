from environment import (
    ComputeTask,
    Clock,
    FixedTime,
    ComputeTaskInfo,
    Message,
    SISOComputeTask,
    MISOFusionTask,
)
import pytest
from unittest.mock import Mock
from typing import Tuple


class DummyComputeTask(ComputeTask):
    def _exec(self, input_args: list[Message]) -> Tuple[Message, ComputeTaskInfo]:
        return input_args[0], ComputeTaskInfo(4, 2), 10


@pytest.fixture
def setup_compute_task():
    clock = Clock(10)
    return (
        clock,
        DummyComputeTask(clock.as_readonly(), FixedTime(10)),
    )


def test_start_cycle_sets_timers(setup_compute_task):
    clock, task = setup_compute_task
    task.run([Message(1, 1, 1), Message(2, 2, 2), Message(3, 3, 3)])
    for _ in range(10):
        assert not task.is_finished
        assert task.is_running
        assert task.t_start == 10
        assert task.duration is None
        assert task.t_end is None
        assert task.result is None
        assert task.compute_info == ComputeTaskInfo(4, 2)
        clock.advance(1)

    assert task.is_finished
    assert not task.is_running
    assert task.t_start == 10
    assert task.duration == 10
    assert task.t_end == 20
    assert task.result == Message(1, 1, 1)
    assert task.compute_info == ComputeTaskInfo(4, 2)


def test_run_twice_raises(setup_compute_task):
    clock, task = setup_compute_task
    task.run([1, 2, 3])
    with pytest.raises(RuntimeError):
        task.run([2, 3, 4])


@pytest.fixture
def setup_siso_compute_task():
    clock = Clock(10)
    sampler = FixedTime(10)
    return clock, SISOComputeTask(clock.as_readonly(), sampler)


def test_siso_input_none(setup_siso_compute_task):
    clock, task = setup_siso_compute_task
    task.run([None])
    assert task.is_finished
    assert task.compute_info == ComputeTaskInfo(1, 1)
    assert task.duration == 0
    assert task.result is None


def test_siso_input_regular(setup_siso_compute_task):
    clock, task = setup_siso_compute_task
    task.run([Message(1, 2, 3)])
    assert task.is_running
    assert task.compute_info == ComputeTaskInfo(1, 0), 10
    clock.advance(10)
    assert task.is_finished
    assert task.t_start == 10
    assert task.duration == 10
    assert task.t_end == 20
    assert task.result.min_time == 1
    assert task.result.avg_time == 2
    assert task.result.max_time == 3


def test_siso_input_invalid_raises(setup_siso_compute_task):
    clock, task = setup_siso_compute_task
    with pytest.raises(RuntimeError):
        task.run([])
    with pytest.raises(RuntimeError):
        task.run([1, 2])


@pytest.fixture
def setup_miso():
    clock = Clock(10)
    sampler = FixedTime(10)
    return clock, MISOFusionTask(clock.as_readonly(), sampler, 5)


def test_miso_all_none(setup_miso):
    clock, task = setup_miso
    task.run([None, None])
    assert task.is_finished
    assert task.result is None
    assert task.compute_info == ComputeTaskInfo(2, 2)
    assert task.duration == 0


def test_miso_empty_raises(setup_miso):
    clock, task = setup_miso
    with pytest.raises(RuntimeError):
        task.run([])


def test_miso_single(setup_miso):
    clock, task = setup_miso
    task.run([Message(1, 2, 3)])
    assert task.is_running
    assert task.compute_info == ComputeTaskInfo(1, 0)

    clock.advance(10)
    assert task.is_finished
    assert task.duration == 10
    assert task.result == Message(1, 2, 3)


def test_miso_multiple_all_accepted(setup_miso):
    clock, task = setup_miso
    task.run([Message(6, 8, 10), Message(5, 6, 9), Message(9, 9, 9), Message(5, 5, 5)])
    clock.advance(10)
    assert task.compute_info == ComputeTaskInfo(4, 0)
    assert task.result == Message(5, (8 + 6 + 9 + 5) / 4, 10)


def test_miso_multiple_some_none(setup_miso):
    clock, task = setup_miso
    task.run([Message(6, 8, 10), None, Message(9, 9, 9), None])
    clock.advance(10)
    assert task.compute_info == ComputeTaskInfo(4, 2)
    assert task.result == Message(6, (8 + 9) / 2, 10)


def test_miso_some_rejected(setup_miso):
    clock, task = setup_miso
    task.run([Message(6, 8, 10), Message(4, 6, 9), Message(9, 9, 9), Message(4, 5, 5)])
    clock.advance(10)
    assert task.compute_info == ComputeTaskInfo(4, 2)
    assert task.result == Message(6, (8 + 9) / 2, 10)


def test_miso_all_rejected(setup_miso):
    clock, task = setup_miso
    task.run([Message(3, 8, 10), Message(3, 6, 9), Message(3, 9, 9), Message(3, 5, 5)])
    assert task.is_finished
    assert task.compute_info == ComputeTaskInfo(4, 4)
    assert task.result is None


def test_miso_reset(setup_miso):
    clock, task = setup_miso
    task.run([Message(1, 2, 3)])
    clock.advance(1)
    assert task.is_running
    assert task.compute_info == ComputeTaskInfo(1, 0)

    task.reset()
    assert task.t_start == clock.get_time_ms()
    assert task.t_end == clock.get_time_ms()
    assert task.is_finished
    assert not task.is_running
    assert task.result == None
    assert task.compute_info == None
