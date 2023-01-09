import torch


class ReplayMemory(object):
    def __init__(self, state_dim, action_dim, capacity, device) -> None:
        self.__device = device
        self.__capacity = capacity
        self.__size = 0
        self.__pos = 0

        self.__m_state = torch.zeros((capacity, state_dim), dtype=torch.float)
        self.__m_next_state = torch.zeros((capacity, state_dim), dtype=torch.float)
        self.__m_actions = torch.zeros((capacity, action_dim), dtype=torch.float)
        self.__m_rewards = torch.zeros((capacity, 1), dtype=torch.float)
        self.__m_dones = torch.zeros((capacity, 1), dtype=torch.bool)

    def push(self, state, next_state, action, reward, done):
        self.__m_state[self.__pos] = state
        self.__m_next_state[self.__pos] = next_state
        self.__m_actions[self.__pos] = action
        self.__m_rewards[self.__pos, 0] = reward
        self.__m_dones[self.__pos, 0] = done

        self.__pos += 1
        self.__size = max(self.__size, self.__pos)
        self.__pos %= self.__capacity

    def sample(self, batch_size):
        indices = torch.randint(0, high=self.__size, size=(batch_size,))
        b_state = self.__m_state[indices].to(self.__device).float()
        b_next = self.__m_next_state[indices].to(self.__device).float()
        b_action = self.__m_actions[indices].to(self.__device)
        b_reward = self.__m_rewards[indices].to(self.__device).float()
        b_done = self.__m_dones[indices].to(self.__device).float()
        return b_state, b_action, b_reward, b_next, b_done

    def __len__(self) -> int:
        return self.__size
