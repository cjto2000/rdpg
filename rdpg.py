import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from rdpg_replay_mem import History

import os
import numpy as np

from rdpg_constants import *
from rdpg_models import Actor, Critic

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

class Orn_Uhlen:
    def __init__(self, n_actions, mu=0, theta=0.15, sigma=0.2):
        self.n_actions = n_actions
        self.X = np.ones(n_actions) * mu
        self.mu = mu
        self.sigma = sigma
        self.theta = theta

    def reset(self):
        self.X = np.ones(self.n_actions) * self.mu

    def sample(self):
        dX = self.theta * (self.mu - self.X)
        dX += self.sigma * np.random.randn(self.n_actions)
        self.X += dX
        return self.X

class RDPG:
    def __init__(self, env, memory):
        self.env = env

        self.state_dim = env.state_dim
        self.action_dim = env.n_actions
        a_limit = env.limit

        self.actor_net = Actor(self.state_dim, self.action_dim, a_limit).to(device)
        self.critic_net = Critic(self.state_dim, self.action_dim).to(device)

        self.target_actor_net = Actor(self.state_dim, self.action_dim, a_limit).to(device)
        self.target_critic_net = Critic(self.state_dim, self.action_dim).to(device)

        self.target_actor_net.load_state_dict(self.actor_net.state_dict())
        self.target_critic_net.load_state_dict(self.critic_net.state_dict())

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=A_LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=C_LEARNING_RATE)

        self.memory = memory
        self.noise = Orn_Uhlen(self.action_dim)

    def train_one_episode(self, batch_size=32):
        S = self.env.reset()
        A = self.env.sample_action()
        S_prime, R, is_done = self.env.take_action(A)
        R_total = 0
        history = History(self.state_dim, self.action_dim)
        history.append(S, S_prime, A, R) # seed for RNN
        S = S_prime
        t = 0
        n = 0
        hidden_states = None
        n_steps = 0
        for _ in range(THRESHOLD_STEPS):
            n_steps += 1
            #self.env.render()
            H_var, S_var = history.get()
            H_var = Variable(torch.FloatTensor(H_var)).unsqueeze(0).to(device) # H_var has shape (1, 1, 28)
            assert(H_var.shape == (1, 1, 28))
            S_var = Variable(torch.FloatTensor(S_var).unsqueeze(0).to(device))
            A_pred, hidden_states = self.actor_net(H_var, S_var, hidden_states) # A_pred has shape (1, 1, 4)
            assert(A_pred.shape == (1, 1, 4))
            A_pred = A_pred.detach()[-1][-1]
            noise = self.noise.sample()
            assert(A_pred.shape == noise.shape)
            A = A_pred.data.cpu().numpy() + noise
            S_prime, R, is_done = self.env.take_action(A)
            R_total += R
            history.append(S, S_prime, A, R)
            if t == LENGTH - 2: # minus 2 because each history starts with a seed or a previous state
                self.memory.append(history)
                history = History(self.state_dim, self.action_dim)
                history.append(S, S_prime, A, R)
                n += 1
            if is_done:
                break
            else:
                S = S_prime
                t = (t + 1) % (LENGTH - 1)  # minus 1 because each history starts with a seed state

        self.memory.set_indices()
        H_batch = self.memory.sample_histories() # H_batch is (BATCH_SIZE, LENGTH, 28)
        assert(H_batch.shape == (BATCH_SIZE, LENGTH, 28))
        R_batch = self.memory.sample_rewards() # R_batch is (BATCH_SIZE, LENGTH, 1)
        assert(R_batch.shape == (BATCH_SIZE, LENGTH, 1))
        A_batch = self.memory.sample_actions() # A_batch is (BATCH_SIZE, LENGTH, 4)
        assert(A_batch.shape == (BATCH_SIZE, LENGTH, 4))
        O_batch = self.memory.sample_observations() # O_batch is (BATCH_SIZE, LENGTH, 24)
        assert(O_batch.shape == (BATCH_SIZE, LENGTH, 24))

        H_batch = torch.from_numpy(H_batch).float().to(device)

        R_batch = torch.from_numpy(R_batch).float().to(device)
        A_batch = torch.from_numpy(A_batch).float().to(device)
        O_batch = torch.from_numpy(O_batch).float().to(device)

        A_critic, hidden_states = self.target_actor_net(H_batch[:, 1:], O_batch[:, 1:]) # A_critic should be (BATCH_SIZE, LENGTH - 1, 4)
        assert(A_critic.shape == (BATCH_SIZE, LENGTH - 1, 4))
        A_critic = A_critic.detach()
        Q_Spr_A, hidden_states = self.target_critic_net(H_batch[:, 1:], A_critic) # Q_Spr_A should be (BATCH_SIZE, LENGTH - 1, 1)
        assert(Q_Spr_A.shape == (BATCH_SIZE, LENGTH - 1, 1))
        Q_Spr_A = Q_Spr_A.detach()

        # Compute target values for each episode
        target_y = R_batch[:, :-1] + GAMMA * Q_Spr_A # y_t = r_t +  gamma * Q(h_{t + 1}, u(h_{t + 1}))
        assert(target_y.shape == (BATCH_SIZE, LENGTH - 1, 1))

        # Compute estimated values
        y, hidden_states = self.critic_net(H_batch[:, :-1], A_batch[:, :-1])
        assert(y.shape == (BATCH_SIZE, LENGTH - 1, 1))

        # Update critic
        critic_loss = torch.mean(torch.pow(target_y - y, 2))
        self.critic_optimizer.zero_grad() # zeros the gradients for backprop
        critic_loss.backward() # add to gradients
        self.critic_optimizer.step() # backprop

        # Update actor
        A_actor, hidden_states = self.actor_net(H_batch, O_batch)
        assert(A_actor.shape == (BATCH_SIZE, LENGTH, 4))
        actor_loss = -1 * torch.mean(self.critic_net(H_batch, A_actor)[0])
        self.actor_optimizer.zero_grad() # zeros the gradients for backprop
        actor_loss.backward() # add to gradients
        self.actor_optimizer.step() # backprop

        self.soft_update()
        self.noise.reset()
        return critic_loss, actor_loss, R_total, n_steps


    def soft_update(self):
        for target, src in zip(self.target_actor_net.parameters(), self.actor_net.parameters()):
            target.data.copy_(target.data * (1.0 - TAU) + src.data * TAU)

        for target, src in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
            target.data.copy_(target.data * (1.0 - TAU) + src.data * TAU)
