from replay_memory import ReplayMemory
import torch.nn as nn
import torch.nn.functional as F
import torch
import gym
import random
from collections import namedtuple, deque
import numpy as np
import time
import matplotlib.pyplot as plt


# TODO LIST
    # TODO: plot q value as a function of iteration on a number on heldout transitions
    # TODO: experiment with having longer epsilon decay duration
REPLAY_MEMORY_SIZE = 8000  # TODO: would the network benefit from a small replay memory so that it forgets transitions from when it was bad?
FINAL_EXPLORATION_FRAME = REPLAY_MEMORY_SIZE
REPLAY_START_SIZE = REPLAY_MEMORY_SIZE/20
TARGET_NETWORK_UPDATE_FREQUENCY = REPLAY_MEMORY_SIZE/10
DISCOUNT_FACTOR = 0.99
EPSILON_INITIAL = 1  # TODO: is epsilon-greedy benefitial in a deterministic environment
EPSILON_FINAL = 0.1
BATCH_SIZE = 32
TOTAL_TRAINING_ITERATIONS = REPLAY_MEMORY_SIZE * 50
UPDATE_FREQUENCY = 4

NUM_INPUT_CHANNELS = 4
KERNEL1 = 8
STRIDE1 = 4
KERNEL2 = 4
STRIDE2 = 2

ACTION_SPACE_SIZE = 2
OBSERVATION_SPACE_SIZE = 4

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


Transition = namedtuple('Transition',
                       ('state', 'action', 'reward', 'next_state', 'terminal'))


class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(NUM_INPUT_CHANNELS, 16, KERNEL1, STRIDE1)  # 16 8x8 filters
        self.conv2 = nn.Conv2d(16, 32, KERNEL2, STRIDE2)  # 32 4x4 filters
        self.linear = nn.Linear(2592, 256)  # 256 units
        self.output_layer = nn.Linear(256, ACTION_SPACE_SIZE)
    
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
        # torch.nn.init.uniform_(self.linear1.weight, a=-1, b=1)
        self.linear2 = nn.Linear(150, 120)
        # torch.nn.init.uniform_(self.linear2.weight, a=-1, b=1)
        self.output_layer = nn.Linear(120, ACTION_SPACE_SIZE)
        # torch.nn.init.uniform_(self.output_layer.weight, a=-1, b=1)
    
    def forward(self, x):
        out1 = F.relu(self.linear1(x))
        out2 = F.relu(self.linear2(out1))
        y = self.output_layer(out2)
        return y


def get_epsilon(iter_num):
    if iter_num >= FINAL_EXPLORATION_FRAME:
        return EPSILON_FINAL
    else:
        return ((EPSILON_FINAL - EPSILON_INITIAL)/FINAL_EXPLORATION_FRAME)*iter_num + EPSILON_INITIAL


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
            with torch.no_grad():
                targets[j] += DISCOUNT_FACTOR*torch.max(optimal_dqn(sample[j].next_state))

    return (preds, targets)


def get_validation_states(env, num_states=500):
    """Collect a set of 1000 states (by taking random actions) to track performance."""
    validation_states = np.zeros((num_states, OBSERVATION_SPACE_SIZE))
    done = True
    validation_states[0] = env.reset()
    for i in range(num_states):
        if done:
            state = env.reset()
        else:
            state, _, done, _ = env.step(random.randint(0, ACTION_SPACE_SIZE - 1))
        validation_states[i] = state
    return torch.tensor(validation_states, dtype=torch.float, device=DEVICE)


def prepopulate_replay_memory(replay_memory, env):
    count = 0
    while count < REPLAY_START_SIZE:
        state = env.reset()
        state = preprocess(state)

        done = False
        while not done:
            random_action = random.randint(0, ACTION_SPACE_SIZE-1)
            next_state, reward, done, _ = env.step(random_action)
            next_state = preprocess(next_state)
            replay_memory.push(Transition(state, random_action, reward, next_state, done))
            count += 1
        

def clip(reward):
    if reward > 0:
        return 1
    elif reward < 0:
        return -1
    else:
        return 0


def train():
    dqn = DQN_dummy()
    optimal_dqn = DQN_dummy()
    optimal_dqn.load_state_dict(dqn.state_dict())
    dqn.to(DEVICE)
    optimal_dqn.to(DEVICE)
    optimal_dqn.eval()

    env = gym.make("CartPole-v1")
    replay_memory = ReplayMemory(REPLAY_MEMORY_SIZE)
    prepopulate_replay_memory(replay_memory, env)

    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(dqn.parameters(), lr=0.00025)
    
    ep_reward_history = []
    avg_reward_values = []
    avg_qs = []

    validation_states = get_validation_states(env)

    total_iterations = 0
    ep = 0
    print(f"Training for {TOTAL_TRAINING_ITERATIONS} iterations...")
    while total_iterations < TOTAL_TRAINING_ITERATIONS:
        visualize_progress = False # (ep % 100) < 5
        state = env.reset()
        done = False
        ep_len = 0
        ep_reward = 0
        while not done:
            state = preprocess(state)
            if random.random() <= get_epsilon(total_iterations):
                action = random.randint(0, ACTION_SPACE_SIZE-1)
            else:
                with torch.no_grad():
                    action = torch.argmax(dqn(state)).item()
            
            if visualize_progress:
                env.render()
                time.sleep(0.01)

            next_state, reward, done, _ = env.step(action)
            reward = clip(reward)
            next_state = preprocess(next_state)
            replay_memory.push(Transition(state, action, reward, next_state, done))
            ep_reward += reward

            if total_iterations % UPDATE_FREQUENCY == 0:
                preds, targets = get_bellman_preds_targets(replay_memory, dqn, optimal_dqn)

                loss = criterion(preds, targets)
                optimizer.zero_grad()
                loss.backward()
                for param in dqn.parameters():
                    param.grad.data.clamp_(-1,1)
                optimizer.step()

            if total_iterations % TARGET_NETWORK_UPDATE_FREQUENCY == 0:
                optimal_dqn.load_state_dict(dqn.state_dict())
                optimal_dqn.eval()

            ep_len += 1
            total_iterations += 1

        ep_reward_history.append(ep_reward)
        if ep % 100 == 0:
            plt.clf()
            plt.plot(ep_reward_history)
            plt.savefig("reward_hist.png")

        # LOG EPISODE STATISTICS
        if ep % 10 == 0 or visualize_progress:
            print(f"EPISODE # {ep}")
            print("\t-Episode length: ", ep_len)
            print("\t-Total iterations", total_iterations)
            # print("\t-Replay memory size: ", len(replay_memory))
            print("\t-Current epsilon: ", get_epsilon(total_iterations))
            avg_reward = sum(ep_reward_history[-100:])/len(ep_reward_history[-100:])
            print("\t-Avg Reward: ", avg_reward)
            with torch.no_grad():
                validation_preds, _ = torch.max(dqn(validation_states), dim=1)
            print("\t-Validation Q: ", torch.mean(validation_preds).item())
            avg_reward_values.append(avg_reward)
            avg_qs.append(validation_preds)
        ep += 1

    env.close()
    np.save("avg_reward.npy", np.array(avg_reward_values))
    np.save("avg_qs.npy", np.array(avg_qs))
    plt.close()

if __name__ == "__main__":
    train()
