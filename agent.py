import random

import torch
import torch.nn.functional as F
import torch.optim as optim
from memory import ReplayMemory
from model import Actor, Critic
from utils import OUNoise


class DDPGAgent(object):

    def __init__(
            self,
            state_dim,
            action_dim,
            device,
            gamma,
            tau,
            seed,
            noise
    ) -> None:
        self.__state_dim = state_dim
        self.__action_dim = action_dim
        self.__device = device
        self.__gamma = gamma
        self.__tau = tau

        self.__r = random.Random()
        self.__r.seed(seed)
        self.__noise = noise

        self.__a_eval = Actor(state_dim, action_dim, device).to(device)
        self.__a_target = Actor(state_dim, action_dim, device).to(device)
        self.__c_eval = Critic(state_dim, action_dim, device).to(device)
        self.__c_target = Critic(state_dim, action_dim, device).to(device)
        self.__optim_a = optim.Adam(self.__a_eval.parameters(), lr=0.0005)
        self.__optim_c = optim.Adam(self.__c_eval.parameters(), lr=0.001)
        self.__mse_loss = torch.nn.MSELoss()

    def choose_action(self, state, training: bool = False, testing: bool = False):
        if training:
            action = self.__a_eval(state.detach().to(self.__device)) + torch.tensor(self.__noise.sample()).to(
                self.__device)
            self.__noise.update()
            return action
        return self.__a_eval(state.to(self.__device))

    def learn(self, memory: ReplayMemory, batch_size: int):
        state_batch, action_batch, reward_batch, next_batch, done_batch = memory.sample(batch_size)

        targets = reward_batch + self.__gamma * (1 - done_batch) * self.__c_target(next_batch, self.__a_target(next_batch))
        c_loss = F.smooth_l1_loss(self.__c_eval(state_batch, action_batch), targets.detach())
        self.__optim_c.zero_grad()
        c_loss.backward()
        self.__optim_c.step()

        a_loss = - self.__c_eval(state_batch, self.__a_eval(state_batch)).mean()
        self.__optim_a.zero_grad()
        a_loss.backward()
        self.__optim_a.step()

    def sync(self):
        for eval_params, target_params in zip(self.__a_eval.parameters(), self.__a_target.parameters()):
            target_params.data.copy_(target_params.data * (1.0 - self.__tau) + eval_params.data * self.__tau)
        for eval_params, target_params in zip(self.__c_eval.parameters(), self.__c_target.parameters()):
            target_params.data.copy_(target_params.data * (1.0 - self.__tau) + eval_params.data * self.__tau)

    def save(self, path: str):
        torch.save(self.__a_eval.state_dict(), path)  # 保存policy网络参数

class DPGAgent() :
    def __init__(
            self,
            state_dim,
            action_dim,
            device,
            gamma,
            tau,
            seed,
            noise
    ) -> None:
        self.__state_dim = state_dim
        self.__action_dim = action_dim
        self.__device = device
        self.__gamma = gamma
        self.__tau = tau

        self.__r = random.Random()
        self.__r.seed(seed)
        self.__noise = noise

        self.__a_eval = Actor(state_dim, action_dim, device).to(device)
        self.__c_eval = Critic(state_dim, action_dim, device).to(device)
        self.__optim_a = optim.Adam(self.__a_eval.parameters(), lr=0.0001)
        self.__optim_c = optim.Adam(self.__c_eval.parameters(), lr=0.001)
        self.__mse_loss = torch.nn.MSELoss()

    def choose_action(self, state, training: bool = False, testing: bool = False):
        if training :
            action = self.__a_eval(state.detach().to(self.__device)) + torch.tensor(self.__noise.sample()).to(self.__device)
            self.__noise.update()
            return action
        return self.__a_eval(state.to(self.__device))

    def learn(self, memory: ReplayMemory, batch_size: int):
        state_batch, action_batch, reward_batch, next_batch, done_batch = memory.sample(batch_size)

        targets = reward_batch + self.__gamma * (1 - done_batch) * self.__c_eval(next_batch, self.__a_eval(next_batch))
        c_loss = F.smooth_l1_loss(self.__c_eval(state_batch, action_batch), targets.detach())
        self.__optim_c.zero_grad()
        c_loss.backward()
        self.__optim_c.step()

        a_loss = - self.__c_eval(state_batch, self.__a_eval(state_batch)).mean()
        self.__optim_a.zero_grad()
        a_loss.backward()
        self.__optim_a.step()

    def sync(self):
        pass

    def save(self, path: str):
        torch.save(self.__a_eval.state_dict(), path)  # 保存policy网络参数
