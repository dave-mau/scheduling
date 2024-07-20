from .interfaces import StateVariableNormalizer


class ConstantNormalizer(StateVariableNormalizer):
    def __init__(self, time_constant: float):
        self._time_constant = time_constant

    def normalize(self, value: float) -> float:
        return value / self._time_constant
