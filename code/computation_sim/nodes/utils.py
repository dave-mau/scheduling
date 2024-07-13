from typing import Tuple

from computation_sim.basic_types import Header, Time
from computation_sim.time import as_age


def header_to_state(header: Header, now: Time) -> Tuple[float, float, float]:
    return [
        float(as_age(header.t_measure_oldest, now)),
        float(as_age(header.t_measure_youngest, now)),
        float(as_age(header.t_measure_average, now)),
    ]
