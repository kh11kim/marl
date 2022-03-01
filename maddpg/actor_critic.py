import torch
import torch.nn as nn
import torch.nn.functional as F


# define the actor network
class Actor(nn.Module):
    def __init__(self, args, agent_id):
        super(Actor, self).__init__()
        self.max_action = 1.
        self.hidden_layer = args.hidden_layer
        self.fc1 = nn.Linear(args.obs_shape[agent_id], self.hidden_layer)
        self.fc2 = nn.Linear(self.hidden_layer, self.hidden_layer)
        self.fc3 = nn.Linear(self.hidden_layer, self.hidden_layer)
        self.action_out = nn.Linear(self.hidden_layer, args.action_shape[agent_id])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.max_action = 1.
        self.hidden_layer = args.hidden_layer
        self.fc1 = nn.Linear(sum(args.obs_shape) + sum(args.action_shape), self.hidden_layer)
        self.fc2 = nn.Linear(self.hidden_layer, self.hidden_layer)
        self.fc3 = nn.Linear(self.hidden_layer, self.hidden_layer)
        self.q_out = nn.Linear(self.hidden_layer, 1)

    def forward(self, state, action):
        state = torch.cat(state, dim=1)
        for i in range(len(action)):
            action[i] /= self.max_action
        action = torch.cat(action, dim=1)
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value
