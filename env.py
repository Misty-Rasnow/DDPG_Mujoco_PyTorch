import numpy as np
import torch
import gym

from agent import *
from utils import batch_normal

class MyEnv(object):

    def __init__(self, device) -> None:
        self.__env_train = gym.make('Walker2d-v2')
        self.__env_test = gym.make('Walker2d-v2')
        self.__env = self.__env_train
        self.__device = device
        self.__low = self.__env.action_space.low
        self.__high = self.__env.action_space.high
        self.__bn = batch_normal(self.get_state_dim())

    def reset(self, render=False):
        state = self.__env.reset()
        self.__bn.update(state[np.newaxis, :])
        state = self.__bn.normal(state)
        return torch.Tensor(state)

    def step(self, action):
        action = self.__low + (self.__high - self.__low) * (action + 1) / 2
        next_state, reward, done, info = self.__env.step(action)
        self.__bn.update(next_state[np.newaxis, :])
        next_state = self.__bn.normal(next_state)
        return torch.Tensor(next_state), reward, done

    def get_state_dim(self):
        return self.__env.observation_space.shape[0]

    def get_action_dim(self):
        return self.__env.action_space.shape[0]

    def evaluate(self, agent, num_episode = 3, render = False):
        self.__env = self.__env_test
        ep_rewards = []
        frames = []
        for _ in range(num_episode):
            state = self.reset(render=render)
            done = False
            ep_reward = 0
            while not done:
                if render :
                    frame = self.__env.render(mode = "rgb_array")
                    frames.append(frame)
                action = agent.choose_action(state, testing=True)
                action = action.cpu().detach().numpy()
                next_state, reward, done = self.step(action)
                ep_reward += reward
                state = next_state

            ep_rewards.append(ep_reward)

        self.__env = self.__env_train
        return np.sum(ep_rewards) / num_episode, frames
