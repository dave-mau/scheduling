from typing import Tuple

from computation_sim.basic_types import Header, Time
from computation_sim.time import as_age

from .interfaces import StateVariableNormalizer
from .state_normalizers import ConstantNormalizer


def header_to_state(
    header: Header,
    now: Time,
    age_normalizer: StateVariableNormalizer = ConstantNormalizer(1.0),
    count_normalizer: StateVariableNormalizer = ConstantNormalizer(1.0),
) -> Tuple[float, float, float, float]:
    return [
        age_normalizer.normalize(float(as_age(header.t_measure_oldest, now))),
        age_normalizer.normalize(float(as_age(header.t_measure_youngest, now))),
        age_normalizer.normalize(float(as_age(header.t_measure_average, now))),
        count_normalizer.normalize(float(header.num_measurements)),
    ]


def empty_message_state(
    age_normalizer: StateVariableNormalizer = ConstantNormalizer(1.0),
    count_normalizer: StateVariableNormalizer = ConstantNormalizer(1.0),
) -> Tuple[float, float, float, float]:
    return [
        age_normalizer.normalize(0.0),
        age_normalizer.normalize(0.0),
        age_normalizer.normalize(0.0),
        count_normalizer.normalize(0.0),
    ]
