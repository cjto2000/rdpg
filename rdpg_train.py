import torch

import pickle
import os

from environment import Game
from rdpg_replay_mem import ReplayMemory, History
from rdpg import RDPG
from rdpg_constants import *

# gam environment
env = Game()

state_dim = env.get_state_dim()
action_dim = env.get_action_dim()

# Create replay memory
memory = ReplayMemory(state_dim, action_dim)

def initialize_replay_mem():
    '''
    Initialize the replay memory.
    '''
    print("Initializing replay memory...")
    S = env.reset()
    n = 0
    t = 0
    history = History(state_dim, action_dim)
    while n < MAX_CAPACITY:
        if n % 2000 == 0:
            print(n)
        A = env.sample_action()
        S_prime, R, is_done = env.take_action(A)
        history.append(S, S_prime, A, R)
        if t == LENGTH - 1:
            memory.append(history)
            history = History(state_dim, action_dim)
            n += 1
        if is_done:
            S = env.reset()
        else:
            S = S_prime
        t = (t + 1) % LENGTH

if os.path.exists("mem.pkl"):
    with open("mem.pkl", "rb") as f:
        memory = pickle.load(f)
else:
    initialize_replay_mem()
    with open("mem.pkl", "wb") as f:
        pickle.dump(memory, f)

# DDPG agent
agent = RDPG(env, memory)

history = {
    "rewards" : [],
    "critic_loss" : [],
    "actor_loss": [],
    "steps": [],
}

if __name__ == "__main__":
    running_R = -200
    best_reward = -float("inf")
    total_steps = 0
    for i in range(N_EPISODES):
        l1, l2, R, n_steps = agent.train_one_episode(BATCH_SIZE)
        total_steps += n_steps
        best_reward = max(R, best_reward)
        running_R = 0.9 * running_R + 0.1 * R
        if i % LOG_STEPS == 0:
            history["rewards"].append(running_R)
            history["critic_loss"].append(l1)
            history["actor_loss"].append(l2)
            history['best_reward'] = best_reward
            history['total_steps'] = total_steps
            history["steps"].append(total_steps)
            print("Episode %5d -- Rewards : %.5f -- Losses: %.5f(a)  %.5f(c) -- Best Reward: %.5f -- Total Steps: %d" % (i, running_R, l2, l1, best_reward, total_steps))
        if R > best_reward:
            best_reward = R
            torch.save(agent.actor_net.state_dict(), actor_model_path)
            torch.save(agent.critic_net.state_dict(), critic_model_path)
    print("Training complete....")