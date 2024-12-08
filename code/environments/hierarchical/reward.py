from typing import Dict, List
import numpy as np

from computation_sim.basic_types import Time

class Reward:
    def __init__(
        self,
        cost_message_loss: float = 1.0,
        cost_output_age: float = 1.0,
        cost_activation: float = 1.0,
    ):
        self._cost_message_loss = cost_message_loss
        self._cost_output_age = cost_output_age
        self._cost_activations = cost_activation

    def __call__(self, action: List[int], buffer_overrides: Dict[str, int], missing_inputs: Dict[str, int], output_age: Time) -> float:
        num_overrides_and_missing = sum(buffer_overrides.values()) + sum(missing_inputs.values())
        num_activations = np.count_nonzero(action)
        reward = -self._cost_message_loss * float(num_overrides_and_missing)
        reward -= self._cost_output_age * float(output_age)
        reward -= self._cost_activations * float(num_activations)
        return reward
