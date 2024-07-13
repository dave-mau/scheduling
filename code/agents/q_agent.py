import random
from math import exp

import torch
from gymnasium.spaces import Space
from torch import nn, optim

from .buffer import Memory, Sample
from .q_network import DQN


class DQNActor(object):
    def __init__(
        self,
        num_states,
        num_actions,
        batch_size=128,
        gamma=0.99,
        epsilon_start=0.9,
        epsilon_end=0.05,
        epsilon_decay=1000,
        tau=0.005,
        lr=1.0e-4,
        memory_size=60_000,
        device="cpu",
        **kwargs,
    ):
        self.num_states = num_states
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.gamma = gamma

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.num_states = num_states
        self.num_actions = num_actions
        self.tau = tau
        self.lr = lr

        self.policy_net = DQN(num_states, num_actions).to(device)
        self.target_net = DQN(num_states, num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = Memory(memory_size)
        self.experience_count = 0  # Total experience collected
        self.learn_count = 0  # Number of learning updates

    def load_weights(self, path: str):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(torch.load(path))

    def eval(self):
        self.policy_net.eval()
        self.target_net.eval()

    def get_epsilon(self) -> float:
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * exp(
            -1.0 * self.experience_count / self.epsilon_decay
        )

    def greedy(self, state) -> int:
        with torch.no_grad():
            return self.policy_net(state).argmax()

    def epsilon_greedy(self, state) -> int:
        if random.random() > self.get_epsilon():
            return self.greedy(state).item()
        else:
            return random.randint(0, self.num_actions - 1)

    def push_memory(self, state, action, next_state, reward):
        self.experience_count += 1
        self.memory.push(state, action, next_state, reward)

    def optimize_model(self) -> dict:
        if len(self.memory) < self.batch_size:
            return {}
        self.learn_count += 1

        # Get samples from memory, prepare for learning
        transitions = self.memory.sample(self.batch_size)
        batch = Sample(*zip(*transitions))
        state_batch = torch.cat(batch.s)
        action_batch = torch.cat(batch.a)
        reward_batch = torch.cat(batch.r)
        next_state_batch = torch.cat(batch.s_prime)

        # Evaluate value function
        q_curr = self.policy_net(state_batch).gather(1, action_batch)
        with torch.no_grad():
            q_next = self.target_net(next_state_batch).max(1).values

        # Compute td-error
        q_curr_expected = reward_batch + self.gamma * q_next
        loss = nn.SmoothL1Loss(beta=1.0)(q_curr, q_curr_expected.unsqueeze(1))

        # Update model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target model
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = (1.0 - self.tau) * target_net_state_dict[
                key
            ] + self.tau * policy_net_state_dict[key]
        self.target_net.load_state_dict(target_net_state_dict)

        return {"loss": loss, "epsilon": self.get_epsilon()}
