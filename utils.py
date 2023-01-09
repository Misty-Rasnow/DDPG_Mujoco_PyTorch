import copy
import random
import numpy as np


class OUNoise:
    """Ornstein-Uhlenbeck process."""
    """"
        Project for Udacity Danaodgree in Deep Reinforcement Learning (DRL)
        Code Expanded and Adapted from Code provided by Udacity DRL Team, 2018.
    """
    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2, sigma_min=0.01, sigma_decay=3e-6):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)
        """Resduce  sigma from initial value to min"""
        self.sigma = max(self.sigma_min, self.sigma*self.sigma_decay)

    def update(self):
        self.sigma = max(self.sigma_min, self.sigma - self.sigma_decay)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx

        return self.state

class NormalNoise :
    def __init__(self, size, seed, mu=0.0, sigma=0.2, sigma_min=0.01, sigma_decay=3e-6):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay
        self.seed = random.seed(seed)
        self.size = size
    
    def update(self) :
        self.sigma = max(self.sigma_min, self.sigma - self.sigma_decay)
    
    def sample(self) :
        return np.random.normal(self.mu, self.sigma)

class batch_normal(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, state):
        batch_mean = np.mean(state, axis=0)
        batch_var = np.var(state, axis=0)
        batch_count = state.shape[0]
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        temp_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        temp_var = M2 / tot_count
        temp_count = tot_count
        self.mean = temp_mean
        self.var = temp_var
        self.count = temp_count

    def normal(self, state):
        return np.clip((state - self.mean) / (self.var ** 0.5 + 1e-8), -5, 5)
