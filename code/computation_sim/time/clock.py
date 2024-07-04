from numpy import round
from ..basic_types import Time


def round_to_fixed_point(func):
    """Decorator that rounds input to the nearest time step."""

    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return round(result).astype(Time)

    return wrapper


class Clock:
    def __init__(self, t0: Time):
        self._t0 = t0
        self._time = Time(t0)

    def get_time_ms(self) -> Time:
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
        self._time = Time(self._t0)

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
        return self._clock.get_time_ms()
