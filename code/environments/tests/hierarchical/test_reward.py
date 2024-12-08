import pytest
from environments.hierarchical import Reward
from unittest.mock import Mock


@pytest.fixture
def reward():
    return Reward(cost_message_loss=1.0, cost_output_age=10.0, cost_activation=100.0)

def test_action_reward_happy(reward):
    action = [1, 2, 3]
    buffer_overrides = {'a': 1, 'b': 2}
    missing_inputs = {'c': 3, 'd': 4}
    output_age = 5
    assert pytest.approx(reward(action, buffer_overrides, missing_inputs, output_age)) == - (1.0 * 10 + 10.0 * 5 + 100.0 * 3)

def test_action_reward_empty(reward):
    action = []
    buffer_overrides = {}
    missing_inputs = {}
    output_age = 5
    assert pytest.approx(reward(action, buffer_overrides, missing_inputs, output_age)) == - (5.0 * 10.0)
