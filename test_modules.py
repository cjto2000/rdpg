import pickle
from rdpg_replay_mem import *
from environment import Game


env = Game()

state_dim = env.get_state_dim()
action_dim = env.get_action_dim()

# TEST REPLAY MEM

with open("mem.pkl", "rb") as f:
    mem = pickle.load(f)


# TEST 0: Making sure that none of the values are 0 arrays
def test0():
    for i in mem.histories:
        zeros = np.zeros((LENGTH, state_dim + action_dim)).flatten()
        i = i.flatten()
        assert(np.sum(zeros == i) != LENGTH * (state_dim + action_dim))

    for i in mem.actions:
        zeros = np.zeros((LENGTH, action_dim)).flatten()
        i = i.flatten()
        assert(np.sum(zeros == i) !=  (LENGTH * action_dim))

    for i in mem.rewards:
        zeros = np.zeros((LENGTH, 1)).flatten()
        i = i.flatten()
        assert(np.sum(zeros == i) != LENGTH)

    for i in mem.observations:
        zeros = np.zeros((LENGTH, state_dim)).flatten()
        i = i.flatten()
        assert(np.sum(zeros == i) != LENGTH * state_dim)


if __name__ == "__main__":
    test0()






