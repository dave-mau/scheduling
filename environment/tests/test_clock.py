from environment import Clock, TimeProvider, TimeMs
import pytest


@pytest.fixture
def clock() -> Clock:
    return Clock(10)


def test_clock_advance_by_one(clock):
    assert clock.get_time_ms() == 10
    clock.advance(1)
    assert clock.get_time_ms() == 11


def test_clock_advance_by_zero(clock):
    clock.advance(0)
    assert clock.get_time_ms() == 10


def test_clock_advance_fail(clock):
    with pytest.raises(AssertionError):
        clock.advance(-1)


def test_clock_advance_float(clock):
    clock.advance(2.9)
    assert clock.get_time_ms() == 12


def test_time_provider(clock):
    provider = clock.as_readonly()
    assert provider.time == 10
    clock.advance(100)
    assert provider.time == 110


def test_clock_add(clock):
    clock += 1
    assert clock.get_time_ms() == 11


def test_clock_reset(clock):
    t0 = clock._time
    clock.advance(10)
    assert t0 != clock.get_time_ms()
    clock.reset()
    assert t0 == clock.get_time_ms()
