import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, device):
        super(Actor, self).__init__()
        self.__fc1 = nn.Linear(state_dim, 400)
        self.__fc2 = nn.Linear(400, 300)
        self.__fc3 = nn.Linear(300, action_dim)

        # initialize parameters
        nn.init.uniform_(self.__fc1.weight, -1 / math.sqrt(state_dim), 1 / math.sqrt(state_dim))
        self.__fc1.bias.data.zero_()
        nn.init.uniform_(self.__fc2.weight, -1 / 20, 1 / 20)
        self.__fc2.bias.data.zero_()
        nn.init.uniform_(self.__fc3.weight, -3 * 1e-3, 3 * 1e-3)
        self.__fc3.bias.data.zero_()

        self.__device = device

    def forward(self, state):
        x = F.relu(self.__fc1(state))
        x = F.relu(self.__fc2(x))
        x = F.tanh(self.__fc3(x))

        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super(Critic, self).__init__()
        self.__fc1 = nn.Linear(state_dim, 400)
        self.__fc2 = nn.Linear(400 + action_dim, 300)
        self.__fc3 = nn.Linear(300, 1)

        # initialize parameters
        nn.init.uniform_(self.__fc1.weight, -1 / math.sqrt(state_dim), 1 / math.sqrt(state_dim))
        self.__fc1.bias.data.zero_()
        nn.init.uniform_(self.__fc2.weight, -1 / 20, 1 / 20)
        self.__fc2.bias.data.zero_()
        nn.init.uniform_(self.__fc3.weight, -3 * 1e-4, 3 * 1e-4)
        self.__fc3.bias.data.zero_()

        self.__device = device

    def forward(self, state, a):
        x = F.relu(self.__fc1(state))
        x = torch.cat((x, a), dim=1)
        x = F.relu(self.__fc2(x))
        x = self.__fc3(x)

        return x

