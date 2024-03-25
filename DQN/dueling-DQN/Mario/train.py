import argparse
import os
import torch
from src.env import create_train_env
from src.model import Net
import shutil
import random
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch.nn as nn
import timeit  # 用于测量代码的执行时间。

os.environ['OMP_NUM_THREADS'] = '1'
device = torch.device('cuda:0')


# 对世界关卡和训练超参数通过命令行设置
def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Asynchronous Methods for Deep Reinforcement Learning for 
        Super Mario Bros""")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="complex")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99, help='discount factor for rewards')
    parser.add_argument('--epsilon', type=float, default=1., help='exploration rate')
    parser.add_argument('--max_episodes', type=int, default=50000, help='max_episodes')
    parser.add_argument('--target_update_steps', type=int, default=1000, help='target_net update steps')
    parser.add_argument('--memory_capacity', type=int, default=100000, help='ReplayBuffer size')
    parser.add_argument('--render', type=bool, default=False, help='is Render')
    parser.add_argument('--batch_size', type=int, default=100, help='batch_size of ReplayBuffer')
    parser.add_argument("--save_interval", type=int, default=500, help="Number of steps between savings")
    parser.add_argument("--max_actions", type=int, default=200, help="Maximum repetition steps in test phase")
    parser.add_argument("--log_path", type=str, default="tensorboard/a3c_super_mario_bros")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--load_from_previous_stage", type=bool, default=False,
                        help="Load weight from previous trained stage")
    parser.add_argument("--use_gpu", type=bool, default=True)
    args = parser.parse_args()
    return args


class ReplayBuffer:
    def __init__(self, MEMORY_CAPACITY):
        self.buffer = deque(maxlen=MEMORY_CAPACITY)

    def add(self, state, action, reward, state_, done):
        self.buffer.append((state, action, reward, state_, done))

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size):
        sub_buffer = random.sample(self.buffer, batch_size)
        state, action, reward, state_, done = zip(*sub_buffer)  # *代表解压
        return np.array(state), action, reward, np.array(state_), done


class DQN:
    def __init__(self, GAMMA, LR, EPSILON, num_actions, num_states, batch_size, target_update_steps):
        self.eval_net, self.target_net = Net(num_states, num_actions).to(device), Net(num_states,
                                                                                      num_actions,
                                                                                      ).to(
            device)
        self.learn_count = 0
        self.lr = LR
        self.epsilon = EPSILON
        self.epsilon_decay = 0.999
        self.epsilon_final = 0.01
        self.gamma = GAMMA
        self.action_dim = num_actions
        self.batch_size = batch_size
        self.target_update_step = target_update_steps
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), self.lr)
        self.loss = nn.MSELoss().to(device)

    def choose_action(self, state):
        state = torch.FloatTensor(state).to(device)
        if np.random.random() <= self.epsilon:
            action = np.random.choice(self.action_dim)
        else:
            actions = self.eval_net(state)
            action = actions.argmax().item()
        return action

    def learn(self, replay_buffer, writer):
        if self.learn_count % self.target_update_step == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        state, action, reward, state_, done = replay_buffer.sample(self.batch_size)
        state = torch.FloatTensor(state).to(device)
        action = torch.tensor(action).view(-1, 1).to(device)
        reward = torch.tensor(reward, dtype=torch.float).view(-1, 1).to(device)
        done = torch.tensor(done, dtype=torch.float).view(-1, 1).to(device)
        state_ = torch.FloatTensor(state_).to(device)
        # Double-DQN
        q_value = self.eval_net(state).gather(1, action)
        action_ = self.eval_net(state_).argmax(dim=1).view(-1, 1).to(device)
        max_next_q_value = self.target_net(state_).gather(1, action_.detach())
        q_target = reward + self.gamma * max_next_q_value * (1 - done)
        losses = torch.mean(self.loss(q_value, q_target))
        writer.add_scalar("Losses", losses, self.learn_count)
        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()
        self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon > self.epsilon_final else self.epsilon_final
        self.learn_count += 1


def train(opt):
    torch.manual_seed(123)
    writer = SummaryWriter(opt.log_path)
    env, num_states, num_actions = create_train_env(opt.world, opt.stage, opt.action_type)
    dqn = DQN(opt.gamma, opt.lr, opt.epsilon, num_actions, num_states, opt.batch_size, opt.target_update_steps)
    replay_buffer = ReplayBuffer(opt.memory_capacity)
    for i_episode in range(1, opt.max_episodes, 1):
        state = env.reset()
        sum_reward = 0
        step_count = 0
        while True:
            step_count += 1
            action = dqn.choose_action(state)
            state_, reward, done, info = env.step(action)
            sum_reward += reward
            replay_buffer.add(state, action, reward, state_, done)
            if len(replay_buffer) >= opt.batch_size:
                dqn.learn(replay_buffer, writer)
            if opt.render:
                env.render()
            if done:
                step_count = 0
                writer.add_scalar("rewards", sum_reward, i_episode)
                if info['flag_get']:
                    model_path = "./trained_models/Double_Dueling_DQN_" + "Mario_v0_{}_{}".format(opt.world,
                                                                                                  opt.stage) + ".pth"
                    torch.save(dqn.eval_net.state_dict(), model_path)
                break

            state = state_
        print(step_count)
    writer.close()


if __name__ == '__main__':
    opt = get_args()
    train(opt)
