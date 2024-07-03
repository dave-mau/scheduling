from torch import nn
from torch.nn import functional


class DQN(nn.Module):
    def __init__(self, n_states, n_actions, width=64):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_states, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, n_actions),
        )

    def forward(self, x):
        return self.model(x)
