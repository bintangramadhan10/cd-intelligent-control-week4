import gymnasium as gym
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Neural Network
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Agen DQN
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).to(self.device)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)

            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state)).item()

            target_f = self.model(state).clone().detach()
            target_f[0][action] = target  # FIX IndexError

            output = self.model(state)
            loss = self.criterion(output, target_f)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    episodes = 300
    batch_size = 32
    scores = []

    for e in range(episodes):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = np.reshape(next_state, [1, state_size])
            reward = reward if not done else -10
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                scores.append(time)
                print(f"Episode: {e+1}/{episodes}, Score: {time}, Epsilon: {agent.epsilon:.2f}")
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    # Simpan model
    torch.save(agent.model.state_dict(), "dqn_cartpole.pth")
    print("Model saved as dqn_cartpole.pth")

    # Plot hasil training
    plt.plot(scores)
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Training Performance (CartPole)")
    plt.savefig("training_result.png")
    plt.show()
