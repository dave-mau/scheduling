from computation_sim.basic_types import Time
from numpy import round


def round_to_fixed_point(func):
    """Decorator that rounds input to the nearest time step."""

    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return round(result).astype(Time)

    return wrapper


class Clock:
    def __init__(self, initial_time: Time):
        self._initial_time = initial_time
        self._time = Time(initial_time)

    @property
    def initial_time(self) -> Time:
        """Returns the initial epoch time in milliseconds."""
        return self._initial_time

    def get_time(self) -> Time:
        """Returns the current epoch time in milliseconds."""
        return self._time

    def advance(self, dt: Time = 1):
        """Advances the clock by dt milliseconds."""
        assert dt >= 0, "Cannot advance time by negative increment."
        self._time = self._increment_time(Time(dt))

    def as_readonly(self) -> "TimeProvider":
        """Return a readonly proxy to the clock."""
        return TimeProvider(self)

    def reset(self):
        """Resets the current time to the initial epoch time."""
        self._time = Time(self._initial_time)

    def __iadd__(self, value: Time):
        """Advance current time by value"""
        self.advance(value)
        return self

    @round_to_fixed_point
    def _increment_time(self, dt: Time):
        return self._time + dt


class TimeProvider:
    """A readonly proxy to the clock that gets the current time."""

    def __init__(self, clock: Clock):
        self._clock = clock

    @property
    def time(self) -> Time:
        """Get the current time in milliseconds relative to the last epoch."""
        return self._clock.get_time()


def as_age(stamp: Time, now: Time):
    assert stamp <= now, "Stamp must be in the past."
    return now - stamp
