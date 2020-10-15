import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    """Critic (Q value) network"""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128 + action_size, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        """Build a critic (Q value) network that maps (state, action) -> Q value."""
        hidden = F.relu(self.fc1(state))
        hidden = F.relu(self.fc2(torch.cat((hidden, action), dim=1)))
        out = self.fc3(hidden)
        return out

