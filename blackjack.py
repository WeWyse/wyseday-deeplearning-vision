import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


# Define the policy network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=-1)
        return x


def choose_action(policy_net, state):
    state = state.clone().detach().float()
    action_probs = policy_net(state)
    # action = np.random.choice(len(action_probs.detach().numpy()), p=action_probs.detach().numpy())
    action = np.argmax(action_probs.detach().numpy())
    return action, action_probs[action]


def discount_rewards(rewards, gamma=0.99):
    discounted = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = running_add * gamma + rewards[t]
        discounted[t] = running_add
    return discounted


def train_agent():
    # Initialize environment and parameters
    env = gym.make("Blackjack-v1", render_mode=None)  # Gym's Blackjack environment
    state_dim = 3  # Environment's state size
    action_dim = env.action_space.n  # Environment's action space size
    hidden_dim1 = 128
    hidden_dim2 = 64

    # Hyperparameters
    learning_rate = 1e-3
    gamma = 0.99
    num_episodes = 2000

    # Initialize policy network
    policy_net = PolicyNetwork(state_dim, hidden_dim1, hidden_dim2, action_dim)
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

    # Tracking metrics
    win_rates = []
    for episode in range(num_episodes):
        state = env.reset()[0]
        done = False

        states = []
        actions = []
        rewards = []

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action, action_prob = choose_action(policy_net, state_tensor)
            next_state, reward, done, _, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

        # Compute discounted rewards
        discounted_rewards = discount_rewards(rewards, gamma)

        # Convert to tensors
        states_tensor = torch.tensor(states, dtype=torch.float32)
        actions_tensor = torch.tensor(actions, dtype=torch.int64)
        rewards_tensor = torch.tensor(discounted_rewards, dtype=torch.float32)

        # Compute policy loss
        policy_loss = []
        action_probs = policy_net(states_tensor)
        for i, reward in enumerate(rewards_tensor):
            log_prob = torch.log(action_probs[i][actions_tensor[i]])
            policy_loss.append(-log_prob * reward)
        policy_loss = torch.stack(policy_loss).sum()

        # Perform backpropagation
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        # Calculate metrics
        total_reward = sum(rewards)
        win = 1 if total_reward > 0 else 0
        win_rates.append(win)

        # Logging progress
        if (episode + 1) % 100 == 0:
            accuracy = np.mean(win_rates[-100:]) * 100
            print(
                f"Episode {episode + 1}/{num_episodes}: Total Reward: {total_reward}, Accuracy: {accuracy:.2f}%"
            )

    env.close()
    return policy_net


def render_agent(policy_net):
    env = gym.make("Blackjack-v1")
    total_rewards = []
    draw_count = 0
    loss_count = 0
    win_count = 0

    for idx in range(10):  # Play at least 10 hands
        state = env.reset()[0]
        done = False
        total_reward = 0
        print("-----------------------------------------")
        print(f"   MATCH # {idx+1}")
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            print("state_tensor", state_tensor)
            action, _ = choose_action(policy_net, state_tensor)
            print("action", action, f" = {'STICK' if action == 0 else 'HIT ME'}")
            state, reward, done, truncated, info = env.step(action)
            print("truncated", truncated, "| info", info)
            print("state", state, "| reward", reward, "| done", done)
            total_reward += reward
            input()

        if reward == 0:
            draw_count += 1
        if reward == 1:
            win_count += 1
        if reward == -1:
            loss_count += 1

        total_rewards.append(total_reward)

    print(f"Average Reward over 10 Hands: {np.mean(total_rewards):.2f}")
    print(f"WINS: {win_count} | DRAWS: {draw_count} | LOSSES: {loss_count}")
    env.close()


if __name__ == "__main__":
    trained_policy = train_agent()
    render_agent(trained_policy)
