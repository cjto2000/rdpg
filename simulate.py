import torch
from torch.autograd import Variable
import gym
from rdpg_models import *
from environment import *

env = Game()
agent = Actor(env.state_dim, env.n_actions, env.limit)

try:
    agent.load_state_dict(torch.load("./saved_models/actor.pth"))
except:
    print("No pretrained model found, using random model!!")
    pass

is_done = False
S = env.reset()
A = env.sample_action()
obs, R, is_done = env.take_action(A)
states = None
while not is_done:
    S = Variable(torch.FloatTensor(S))
    x = np.append(S, A)
    x = torch.from_numpy(x).float()
    obs = torch.from_numpy(obs).float()
    action, states = agent.inference(x, obs, states)
    A = action.detach().numpy().squeeze()
    obs, R, is_done = env.take_action(A)
    if is_done:
        S = obs
    env.env.render()
