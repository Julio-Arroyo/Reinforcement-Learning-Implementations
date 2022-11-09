from replay_memory import ReplayMemory
import torch.nn as nn
import torch.nn.functional as F
import torch
import gym
import random
from collections import namedtuple, deque
import numpy as np
import time

# TODO LIST
    # TODO: visualize what the shit this agent is doing
    # TODO: plot q value as a function of iteration on a number on heldout transitions
    # TODO: experiment with having longer epsilon decay duration
CAPACITY = 1000000  # TODO: would the network benefit from a small replay memory so that it forgets transitions from when it was bad?
NUM_INPUT_CHANNELS = 4
KERNEL1 = 8
STRIDE1 = 4
KERNEL2 = 4
STRIDE2 = 2
NUM_ACTIONS = 4
BATCH_SIZE = 32
EPSILON_INITIAL = 1  # TODO: is epsilon-greedy benefitial in a deterministic environment
EPSILON_FINAL = 0.01
EPSILON_DECAY_DURATION = 5000
GAMMA = 0.99
OBSERVATION_SPACE_SIZE = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


Transition = namedtuple('Transition',
                       ('state', 'action', 'reward', 'next_state', 'terminal'))


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
        self.linear1 = nn.Linear(OBSERVATION_SPACE_SIZE, 150)
        self.linear2 = nn.Linear(150, 120)
        self.output_layer = nn.Linear(120, NUM_ACTIONS)
        # TODO: don't i need a softmax layer?
    
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


def preprocess(state):
    # TODO: implement frame-skipping and down sampling image
    return torch.tensor(state, device=DEVICE)


def get_bellman_preds_targets(replay_memory, dqn, optimal_dqn):
    sample = replay_memory.sample(BATCH_SIZE)
    targets = torch.zeros((BATCH_SIZE,), device=DEVICE)  # BATCH_SIZE-dimensional column vector
    preds = torch.zeros((BATCH_SIZE,), device=DEVICE)
    assert len(sample) == BATCH_SIZE

    for j in range(BATCH_SIZE):
        preds[j] = dqn(sample[j].state)[sample[j].action]

        targets[j] = sample[j].reward
        if not sample[j].terminal:
            targets[j] += GAMMA*torch.max(optimal_dqn(sample[j].next_state))

    return (preds, targets)


def train(num_episodes):
    replay_memory = ReplayMemory(CAPACITY)
    dqn = DQN_dummy()
    optimal_dqn = DQN_dummy()
    optimal_dqn.load_state_dict(dqn.state_dict())
    dqn.to(DEVICE)
    optimal_dqn.to(DEVICE)
    optimal_dqn.eval()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(dqn.parameters())
    env = gym.make('LunarLander-v2')
    ep_reward_history = deque(maxlen=100)
    avg_reward_values = []
    # print(gym.envs.registry.all())

    total_iterations = 0
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        ep_len = 0
        ep_reward = 0
        while not done:
            state = preprocess(state)
            if random.random() <= get_epsilon(total_iterations):
                action = random.randint(0, NUM_ACTIONS-1)
            else:
                action = torch.argmax(dqn(state)).item()
            
            if ep % 10 == 0:
                env.render()
                time.sleep(0.01)

            next_state, reward, done, _ = env.step(action)
            next_state = preprocess(next_state)
            replay_memory.push(Transition(state, action, reward, next_state, done))
            ep_reward += reward

            if len(replay_memory.memory) < BATCH_SIZE:
                ep_len += 1
                total_iterations += 1
                continue

            preds, targets = get_bellman_preds_targets(replay_memory, dqn, optimal_dqn)

            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ep_len += 1
            total_iterations += 1

        ep_reward_history.append(ep_reward)

        # LOG EPISODE STATISTICS
        if ep % 10 == 0:
            print(f"EPISODE # {ep}")
            print("\t-Episode length: ", ep_len)
            print("\t-Total iterations", total_iterations)
            print("\t-Replay memory size: ", len(replay_memory))
            print("\t-Current epsilon: ", get_epsilon(total_iterations))
            avg_reward = sum(ep_reward_history)/len(ep_reward_history)
            print("\t-Avg Reward: ", avg_reward)
            avg_reward_values.append(avg_reward)
            optimal_dqn.load_state_dict(dqn.state_dict())

    env.close()
    np.save("avg_reward.npy", np.array(avg_reward_values))

if __name__ == "__main__":
    train(500)
