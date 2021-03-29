import gym
import numpy as np

class Game:
    def __init__(self):
        self.env = gym.make('BipedalWalker-v3')
        self.state_dim = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]
        # typecast since problems with multiplication if type(limit) = np.float32
        self.limit = self.env.action_space.high

    def reset(self):
        start = self.env.reset()
        return start

    def take_action(self, action):
        new_frame, reward, is_done, _ = self.env.step(action)
        return new_frame, reward, is_done

    def sample_action(self):
        return self.env.action_space.sample()

    def render(self):
        self.env.render()

    def get_state_dim(self):
        return self.state_dim

    def get_action_dim(self):
        return self.n_actions
