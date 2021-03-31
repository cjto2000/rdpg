import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    # Actor provides the next action to take
    def __init__(self, state_dim, action_dim, limit, hidden_size=100):
        super().__init__()
        self.limit = torch.FloatTensor(limit)

        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.lstm = nn.LSTM(hidden_size + state_dim, hidden_size, batch_first=True)
        self.fc2 = nn.Linear(hidden_size, action_dim)
        nn.init.uniform_(self.fc2.weight, -0.003, 0.003)

    def forward(self, history, obs, hidden_states=None):
        x = F.relu(self.fc1(history))
        x = torch.cat((x, obs), dim=2)
        x, states = self.lstm(x, hidden_states)
        x = F.tanh(self.fc2(x))
        return x, states

    def inference(self, x, obs, hidden_state=None):
        """
        x is a single dim numpy array (28,)
        - 28 because state is 24 dim and action is 4 dim
        """
        x = x.view(1, -1).unsqueeze(0)
        obs = obs.view(1, -1).unsqueeze(0)
        x, hidden_state = self.forward(x, obs, hidden_state)
        return x, hidden_state


class Critic(nn.Module):
    # Critic estimates the state-value function Q(S, A)
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        nn.init.xavier_uniform_(self.fc1.weight)

        self.fc2 = nn.Linear(400 + action_dim, 300)
        nn.init.xavier_uniform_(self.fc2.weight)

        self.fc3 = nn.Linear(300, 1)
        nn.init.uniform_(self.fc3.weight, -0.003, 0.003)

    def forward(self, state, action):
        s = F.relu(self.fc1(state))
        x = torch.cat((s, action), dim=1) # should dim be 1 or 2?
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

