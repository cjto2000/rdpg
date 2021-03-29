import numpy as np
from collections import deque
from rdpg_constants import *

class ReplayMemory:
    def __init__(self, state_dim, action_dim):
        self.index = 0
        self.histories = np.zeros((MAX_CAPACITY, LENGTH, state_dim + action_dim))
        self.observations = np.zeros((MAX_CAPACITY, LENGTH, state_dim))
        self.actions = np.zeros((MAX_CAPACITY, LENGTH, action_dim))
        self.rewards = np.zeros((MAX_CAPACITY, LENGTH, 1))
        self.indices = None

    def append(self, history):
        '''
        Puts experience into memory buffer
        args:
            :experience: a tuple consisting of (S, A, S_prime, R)
        '''
        states = history.get_states()
        observations = history.get_observations()
        actions = history.get_actions()
        self.histories[self.index] = np.concatenate((states, actions), axis=1)
        self.observations[self.index] = observations
        self.actions[self.index] = actions
        self.rewards[self.index] = history.get_rewards()
        self.index = (self.index + 1) % MAX_CAPACITY

    def set_indices(self):
        self.indices = np.random.choice([i for i in range(MAX_CAPACITY)], BATCH_SIZE)

    """
    set indices must be called before the following functions
    """

    def sample_histories(self):
        return self.histories[self.indices]

    def sample_actions(self):
        return self.actions[self.indices]

    def sample_observations(self):
        return self.observations[self.indices]

    def sample_rewards(self):
        return self.rewards[self.indices]

class History:
    def __init__(self, state_dim, action_dim):
        self.index = 0
        self.states = np.zeros((LENGTH, state_dim))
        self.observations = np.zeros((LENGTH, state_dim))
        self.actions = np.zeros((LENGTH, action_dim))
        self.rewards = np.zeros((LENGTH, 1))
        self.histories = np.zeros((LENGTH, state_dim + action_dim))

    def append(self, obs, obs_prime, action, reward):
        self.states[self.index] = obs
        self.observations[self.index] = obs_prime
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.histories[self.index] = np.concatenate((obs, action))
        self.index += 1

    def get(self):
        return self.histories[self.index - 1:self.index], self.observations[self.index - 1:self.index]

    def get_states(self):
        return self.states

    def get_observations(self):
        return self.observations

    def get_actions(self):
        return self.actions

    def get_rewards(self):
        return self.rewards