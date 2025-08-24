import copy
from collections import deque
import random
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#=========================================================
# 
# DQN ネットワークの構成部分
#
#=========================================================

class QNet(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.l1 = nn.Linear(6, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 64)
        self.l4 = nn.Linear(64, 64)
        self.l5 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = self.l5(x)
        return x

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        self.buffer.append(data)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)

        state = torch.tensor(np.stack([x[0] for x in data]))
        action = torch.tensor(np.array([x[1] for x in data]).astype(np.int64))
        reward = torch.tensor(np.array([x[2] for x in data]).astype(np.float32))
        next_state = torch.tensor(np.stack([x[3] for x in data]))
        done = torch.tensor(np.array([x[4] for x in data]).astype(np.int32))
        return state, action, reward, next_state, done

class DQNAgent:
    def __init__(self):
        self.gamma = 0.999
        self.lr = 0.0005
        self.epsilon = 0.1
        self.buffer_size = 10000
        self.batch_size = 128
        self.action_size = 5

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.qnet = QNet(self.action_size)
        self.qnet_target = QNet(self.action_size)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr) #最適化アルゴリズムとしてAdamを使用

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = torch.tensor(state[np.newaxis, :])
            qs = self.qnet(state)
            return qs.argmax().item() #Q値の高いほうのインデックスを返す

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.get_batch()
        qs = self.qnet(state)
        q = qs[np.arange(len(action)), action]

        next_qs = self.qnet_target(next_state)
        next_q = next_qs.max(1)[0]

        next_q.detach()
        target = reward + (1 - done) * self.gamma * next_q

        loss_fn = nn.MSELoss() #損失関数として平均二乗誤差を使用
        loss = loss_fn(q, target) #Q値とターゲットQ値の誤差を計算

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sync_qnet(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())

episodes = 1001
sync_interval = 20

gym.envs.register(
  id='dq3-v0',
  entry_point='myenv_orb:MyEnv',
  max_episode_steps=500
)

env = gym.make('dq3-v0')

#=========================================================
#
# 実際の学習部分
#
#=========================================================

agent = DQNAgent()
reward_history = []

episode = 0
episodeTotal = 0

wins = 0
orbs = 0
allDone = False

while not allDone:
    state, _ = env.reset()
    env.setLog(episode == 99)
    done = False
    total_reward = 0
    orbF = False
    winFlag = False

    if episode >= 80:
        agent.epsilon = 0
        agent.lr = 0
    else:
        agent.epsilon = 0.1
        agent.lr = 0.005

    while not done:
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            done = True
        agent.update(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if reward > 0.1:
            if episode >= 80:
                wins += 1
            winFlag = True
        if info["orb"] and not orbF:
            orbs += 1
            orbF = True

    if episode % sync_interval == 0:
        agent.sync_qnet()

    reward_history.append(total_reward)
    # print("エピソード", episodeTotal, "の合計報酬:", total_reward, ("Win" if winFlag else "Lose"))

    episode += 1
    episodeTotal += 1
    if episode == 100:
        print("20戦中 勝利数:", wins, "100戦中 ひかりのたま使用数:", orbs)
        if wins > 15:
            print("総エピソード数:", episodeTotal)
            allDone = True
        wins = 0
        orbs = 0
        episode = 0
