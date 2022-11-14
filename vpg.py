"""Vanilla Policy Gradient"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from torch.distributions import Categorical
from collections import deque
import matplotlib.pyplot as plt
import time


DIM_OBS_SPACE = 4
DIM_ACTION_SPACE = 2
DISCOUNT_FACTOR = 0.99
NUM_EPISODES = 1000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class PolicyNet(nn.Module):
    def __init__(self, dim_obs_space, dim_action_space):
        super(PolicyNet, self).__init__()
        hidden_dim = 200
        self.fc1 = nn.Linear(dim_obs_space, hidden_dim)
        self.out = nn.Linear(hidden_dim, dim_action_space)
    
    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        out_probs = F.softmax(self.out(x1), dim=0)
        return out_probs


def generate_episode(policy, env, visualize):
    episode = []
    state = torch.tensor(env.reset(), device=DEVICE)
    done = False

    while not done:
        action_probs = policy(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        if visualize:
            env.render()
            time.sleep(0.01)
            
        next_state, reward, done, _ = env.step(action.item())
        episode.append((state, action, reward))

        state = torch.tensor(next_state, device=DEVICE)
    
    return episode


def get_expected_reward(episode, t):
    """Get future cumulative discounted reward starting at t"""
    expected_reward = 0
    for k in range(t, len(episode)):
        R_k = episode[k][2] # index 2 corresponds to action
        gamma_k = DISCOUNT_FACTOR**(k-t)
        expected_reward += gamma_k*R_k
    return expected_reward


def train():
    policy_net = PolicyNet(DIM_OBS_SPACE, DIM_ACTION_SPACE)
    optimizer = torch.optim.Adam(policy_net.parameters())
    env = gym.make("CartPole-v1")
    total_episode_rewards = deque(maxlen=100)
    average_episode_rewards = []

    for ep_num in range(NUM_EPISODES):
        visualize = ep_num % 250 == 0
        episode = generate_episode(policy_net, env, visualize)

        for t in range(len(episode)):
            s_t = episode[t][0]  # 0-th index is state
            a_t = episode[t][1]  # 1 index is action
            G_t = get_expected_reward(episode, t)
            gamma_t = DISCOUNT_FACTOR**t
            action_probs = policy_net(s_t)
            action_dist = Categorical(action_probs)

            loss = -gamma_t*G_t*action_dist.log_prob(a_t)

            # update policy parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        episode_reward = sum([episode[i][2] for i in range(len(episode))])
        total_episode_rewards.append(episode_reward)

        if ep_num % 10 == 0:
            avg_episode_reward = sum(total_episode_rewards)/len(total_episode_rewards)
            average_episode_rewards.append(avg_episode_reward)
            print("EPISODE #", ep_num)
            print("\t-Avg total reward per episode: ", avg_episode_reward)
    
    plt.plot(average_episode_rewards)
    plt.xlabel("Episode (x10)")
    plt.ylabel("Total reward per episode (last 100 episodes)")
    plt.title("VPG on CartPole-v1")
    plt.savefig("vpg.png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    train()
