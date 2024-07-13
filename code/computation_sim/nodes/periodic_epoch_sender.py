from typing import List

from computation_sim.basic_types import (
    Header,
    Message,
    Time,
)
from computation_sim.time import DurationSampler

from .interfaces import Sensor


class PeriodicEpochSensor(Sensor):
    def __init__(self, epoch: Time, period: Time, disturbance: DurationSampler, **kwargs):
        super().__init__(**kwargs)
        self._epoch = epoch
        self._period = period
        self._disturbance = disturbance
        self._nominal_send_time = self._epoch
        self._actual_send_time = self._nominal_send_time

    @property
    def state(self) -> List[float]:
        return []

    def get_measurement(self) -> Message | None:
        if self.has_measurement:
            result = Message(self._get_header())
            result.data = self._get_data()
            return result
        return None

    def update(self, time: Time):
        super().update(time)
        if time >= self._actual_send_time:
            self._nominal_send_time += self._period
            while self._actual_send_time <= time:
                self._actual_send_time = self._nominal_send_time + self._disturbance.sample()
            self._has_measurement = True
            return
        self._has_measurement = False

    def reset(self):
        super().reset()
        self._nominal_send_time = self._epoch
        self._actual_send_time = self._nominal_send_time

    def _get_header(self) -> Header:
        header = Header(self._last_update_time, self._last_update_time, self._last_update_time)
        return header

    def _get_data(self) -> object:
        return {}
