import gymnasium as gym
import numpy as np
import torch
from dqn_agent import DQNAgent  # pakai agent dari file dqn_agent.py

if __name__ == "__main__":
    # Buat environment dengan visualisasi
    env = gym.make("CartPole-v1", render_mode="human")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Inisialisasi agen & load model hasil training
    agent = DQNAgent(state_size, action_size)
    agent.model.load_state_dict(torch.load("dqn_cartpole.pth"))
    agent.model.eval()   # set ke mode evaluasi
    agent.epsilon = 0.01 # minim eksplorasi saat testing

    # Jalankan agen untuk 5 episode
    for e in range(5):
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            action = agent.act(state)  # pilih aksi dari agen
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = np.reshape(next_state, [1, state_size])
            state = next_state
            if done:
                print(f"Test Episode: {e+1}, Score: {time}")
                break

    env.close()
