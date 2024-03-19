import random

import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
import pandas as pd
import gym
# from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self, STATE_DIM, ACTION_DIM, HIDDEN_LAYERS):
        super(Net, self).__init__()
        self.output_layer = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN_LAYERS),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYERS, HIDDEN_LAYERS),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYERS, ACTION_DIM)
        )

    def forward(self, x):
        return self.output_layer(x)


class ReplayBuffer:
    def __init__(self, MEMORY_CAPACITY):
        self.capacity = MEMORY_CAPACITY
        self.buffer = []

    def clear_buffer(self):
        del self.buffer[:]

    def add(self, state, action, reward, state_, done):
        if len(self.buffer) == self.capacity:
            del self.buffer[0]
        self.buffer.append((state, action, reward, state_, done))

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size):
        sub_buffer = random.sample(self.buffer, batch_size)
        state, action, reward, state_, done = zip(*sub_buffer)  # *代表解压
        return np.array(state), action, reward, np.array(state_), done


class DQN:
    def __init__(self, GAMMA, LR, EPSILON, ACTION_DIM, STATE_DIM, HIDDEN_LAYERS, BATCH_SIZE, TARGET_UPDATE_STEP):
        self.eval_net, self.target_net = Net(STATE_DIM, ACTION_DIM, HIDDEN_LAYERS).to(device), Net(STATE_DIM,
                                                                                                   ACTION_DIM,
                                                                                                   HIDDEN_LAYERS).to(
            device)
        self.learn_count = 0
        self.lr = LR
        self.epsilon = EPSILON
        self.gamma = GAMMA
        self.action_dim = ACTION_DIM
        self.batch_size = BATCH_SIZE
        self.target_update_step = TARGET_UPDATE_STEP
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), self.lr)
        self.loss = nn.MSELoss().to(device)

    def choose_action(self, state):
        state = torch.FloatTensor(state).to(device)
        if np.random.randn() <= self.epsilon:
            actions = self.eval_net(state)
            action = actions.argmax().item()
        else:
            action = np.random.randint(0, self.action_dim)
        return action

    def learn(self, replay_buffer):
        # print(1)
        state, action, reward, state_, done = replay_buffer.sample(self.batch_size)
        state = torch.FloatTensor(state).to(device)
        action = torch.tensor(action).view(-1, 1).to(device)
        reward = torch.tensor(reward, dtype=torch.float).view(-1, 1).to(device)
        done = torch.tensor(done, dtype=torch.float).view(-1, 1).to(device)
        q_value = self.eval_net(state).gather(1, action)
        state_ = torch.FloatTensor(state_).to(device)
        max_next_q_value = self.target_net(state_).max(1)[0].view(-1, 1)
        q_target = reward + self.gamma * max_next_q_value * (1 - done)
        losses = torch.mean(self.loss(q_value, q_target))
        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()
        self.learn_count += 1
        if self.learn_count % self.target_update_step == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

    # def plot(self, step_count_ls):


def main():
    # hyper parameters
    env_name = 'MountainCar-v0'
    MAX_EPISODE = 10000
    GAMMA = 0.99  # discount factor
    LR = 0.001  # learning rate
    EPSILON = 0.9
    MEMORY_CAPACITY = 10000
    BATCH_SIZE = 64
    HIDDEN_LAYERS = 30
    TARGET_UPDATE_STEPS = 20
    env = gym.make(env_name)
    STATE_DIM = env.observation_space.shape[0]
    ACTION_DIM = env.action_space.n
    RENDER = False
    # print(STATE_DIM, ACTION_DIM)
    dqn = DQN(GAMMA, LR, EPSILON, ACTION_DIM, STATE_DIM, HIDDEN_LAYERS, BATCH_SIZE, TARGET_UPDATE_STEPS)
    replay_buffer = ReplayBuffer(MEMORY_CAPACITY)
    step_counter_ls = []
    sum_step_count = 0
    for i_episode in range(1, MAX_EPISODE, 1):
        state = env.reset()
        # sum_reward = 0
        step_count = 0
        while True:
            step_count += 1
            sum_step_count += 1
            action = dqn.choose_action(state)
            # print(action)
            state_, reward, done, _ = env.step(action)
            reward = reward * 100 if reward > 0 else reward * 5  # 放大，便于观察
            replay_buffer.add(state, action, reward, state_, done)
            if len(replay_buffer) >= MEMORY_CAPACITY:
                dqn.learn(replay_buffer)
            if RENDER:
                env.render()
            if done:
                step_counter_ls.append(step_count)
                if len(step_counter_ls) % 20 == 0:
                    sum_step_count /= 20
                    print('Episode={} ,step_count={}, '.format(i_episode, sum_step_count))
                    sum_step_count = 0
                break
            state = state_
    model_path = "./DQN_" + env_name + ".pth"
    torch.save(dqn.target_net.state_dict(),model_path)


if __name__ == '__main__':
    main()
