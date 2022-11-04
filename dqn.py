from replay_memory import ReplayMemory
import torch.nn as nn
import torch.nn.functional as F
import torch
import gym
import random
from collections import namedtuple


CAPACITY = 10000
NUM_INPUT_CHANNELS = 4
KERNEL1 = 8
STRIDE1 = 4
KERNEL2 = 4
STRIDE2 = 2
NUM_ACTIONS = 2
BATCH_SIZE = 32
EPSILON_INITIAL = 1
EPSILON_FINAL = 0.1
EPSILON_DECAY_DURATION = 10000


Transition = namedtuple('Transition',
                       ('state', 'action', 'reward', 'next_state'))


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(NUM_INPUT_CHANNELS, 16, KERNEL1, STRIDE1)  # 16 8x8 filters
        self.conv2 = nn.Conv2d(16, 32, KERNEL2, STRIDE2)  # 32 4x4 filters
        self.linear = nn.Linear(2592, 256)  # 256 units
        self.output_layer = nn.Linear(256, NUM_ACTIONS)
    
    def forward(self, x):
        out1 = F.relu(self.conv1(x))
        print(out1.shape)
        out2 = F.relu(self.conv2(out1))
        print(out2.shape)
        out2 = torch.flatten(out2, 1) # flatten all dimensions except batch
        print(out2.shape)
        out3 = F.relu(self.linear(out2))
        print(out3.shape)
        y = self.output_layer(out3)
        print(y.shape)
        return y

class DQN_dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 100)
        self.linear2 = nn.Linear(100, 100)
        self.output_layer = nn.Linear(100, NUM_ACTIONS)
    
    def forward(self, x):
        out1 = F.relu(self.linear1(x))
        out2 = F.relu(self.linear2(out1))
        y = self.output_layer(out2)
        return y


def get_epsilon(iter_num):
    if iter_num >= EPSILON_DECAY_DURATION:
        return EPSILON_FINAL
    else:
        return ((EPSILON_FINAL - EPSILON_INITIAL)/EPSILON_DECAY_DURATION)*iter_num + EPSILON_INITIAL


def train(num_episodes):
    replay_memory = ReplayMemory(CAPACITY)
    dqn = DQN_dummy()
    # print(gym.envs.registry.all())
    env = gym.make('LunarLander-v2')
    for ep in num_episodes:
        state = env.reset()
        done = False
        i = 0
        while not done:
            epsilon = get_epsilon(i)
            if random.random() <= epsilon:
                action = random.randint(0, 3)
            else:
                action = torch.argmax(DQN_dummy(state))
            
            next_state, reward, done, info = env.step(action)
            replay_memory.push(Transition(state, action, reward, next_state))

            if len(replay_memory.memory) >= BATCH_SIZE:
                # TODO: SAMPLE AND OPTIMIZE
            i += 1


if __name__ == "__main__":
    train(1)
