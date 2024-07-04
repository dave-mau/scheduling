import gymnasium as gym
from gymnasium import spaces
from typing import Callable, List

from .impl.system import System
from .impl.clock import Clock, TimeMs
from .impl.messaging import Message
import numpy as np
from collections import namedtuple

CostConfig = namedtuple(
    "CostConfig",
    (
        "message_loss_counters",
        "message_loss_weights",
        "output_age_weight",
        "input_cost",
    ),
)


class TimeSchedulingEnv(gym.Env):
    def __init__(
        self,
        clock: Clock,
        system: System,
        measurement_time_cost: float,
        input_cost: float,
        message_loss_costs: float,
        dt: TimeMs = 5,
        render_mode=None,
        state_normalization=100,
    ):
        self._clock = clock
        self._system = system
        self._measurement_time_cost = measurement_time_cost
        self._input_cost = input_cost
        self._message_loss_costs = message_loss_costs
        self._dt = dt
        self._render_mode = render_mode
        self._state_normalization = state_normalization

        self.action_space = spaces.Discrete(int((1 - 2**self._system.num_action_nodes) / (1 - 2)))
        self.observation_space = spaces.Box(
            low=np.zeros((self._system.num_states,), dtype=float),
            high=np.concatenate(
                [
                    np.inf * np.ones((self._system.num_state_nodes,)),  # compute_start_ages
                    np.inf * np.ones((self._system.num_state_nodes,)),  # buf_out_min_ages
                    np.inf * np.ones((self._system.num_state_nodes,)),  # buf_out_avg_ages
                    np.inf * np.ones((self._system.num_state_nodes,)),  # buf_out_max_ages
                    np.ones((self._system.num_state_nodes,)),  # compute_running
                    np.ones((self._system.num_state_nodes,)),  # buf_out_has_value
                ],
                axis=0,
                dtype=float,
            ),
            dtype=float,
        )

        self._last_out = None

    @property
    def time(self) -> TimeMs:
        return self._clock.get_time_ms()

    @property
    def last_output_msg(self) -> Message:
        return self._last_out

    @property
    def last_output_min_age(self) -> TimeMs:
        return (self._clock.get_time_ms() - self._last_out.min_time) / self._state_normalization

    @property
    def last_output_avg_age(self) -> TimeMs:
        return (self._clock.get_time_ms() - self._last_out.avg_time) / self._state_normalization

    @property
    def last_output_max_age(self) -> TimeMs:
        return (self._clock.get_time_ms() - self._last_out.max_time) / self._state_normalization

    def reset(self, seed=None):
        super().reset(seed=seed)
        np.random.seed(seed)
        self._clock.reset()
        self._system.reset()
        self._last_out = Message(
            self._clock.get_time_ms(),
            self._clock.get_time_ms(),
            self._clock.get_time_ms(),
        )
        return self._system.get_state(), {}

    def unpack_action(self, action: int) -> np.ndarray:
        assert action >= 0 and action <= 255
        bits = np.unpackbits(np.uint8(action))
        return bits[-self._system.num_action_nodes :]

    def step(self, action: int):
        # Clean-up all counts of messages lost in previous iterations
        self._system.message_loss_counter.reset()

        # Step simulation
        a = self.unpack_action(action)
        self._system.act(a)
        self._clock.advance(self._dt)
        self._system.update()

        # Update output element
        if self._system.state_nodes[-1].output.has_element:
            self._last_out = self._system.state_nodes[-1].output.read()

        # Get state, compute cost
        state = self._system.get_state(age_normalization_factor=self._state_normalization)
        reward = -self._message_loss_costs * self._system.message_loss_counter.total_counts
        reward -= self._measurement_time_cost * self.last_output_min_age
        reward -= self._input_cost * np.sum(a)

        return (
            state,
            reward,
            False,
            False,
            {"total_message_losses": self._system.message_loss_counter.total_counts},
        )
