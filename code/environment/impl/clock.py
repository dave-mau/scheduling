from numpy import round, uint64

TimeMs = uint64


def round_to_milliseconds(func):
    """Decorator that rounds input to the nearest TimeMs."""

    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return round(result).astype(TimeMs)

    return wrapper


class Clock:
    def __init__(self, t0: TimeMs):
        self._t0 = t0
        self._time = TimeMs(t0)

    def get_time_ms(self) -> TimeMs:
        """Returns the current epoch time in milliseconds."""
        return self._time

    def advance(self, dt: TimeMs = 1):
        """Advances the clock by dt milliseconds."""
        assert dt >= 0, "Cannot advance time by negative increment."
        self._time = self._increment_time(TimeMs(dt))

    def as_readonly(self) -> "TimeProvider":
        """Return a readonly proxy to the clock."""
        return TimeProvider(self)

    def reset(self):
        """Resets the current time to the initial epoch time."""
        self._time = TimeMs(self._t0)

    def __iadd__(self, value: TimeMs):
        """Advance current time by value"""
        self.advance(value)
        return self

    @round_to_milliseconds
    def _increment_time(self, dt: TimeMs):
        return self._time + dt


class TimeProvider:
    """A readonly proxy to the clock that gets the current time."""

    def __init__(self, clock: Clock):
        self._clock = clock

    @property
    def time(self) -> TimeMs:
        """Get the current time in milliseconds relative to the last epoch."""
        return self._clock.get_time_ms()
