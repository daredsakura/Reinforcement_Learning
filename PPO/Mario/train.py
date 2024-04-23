# import os
import argparse
import multiprocessing as mp
import torch
from src.env import create_env
from src.model import Net
# import shutil
# import random
# import torch.nn.functional as F
# from torch.distributions import Categorical
# from collections import deque
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import torch.nn as nn

# import timeit  # 用于测量代码的执行时间。

device = torch.device('cuda:0')


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Proximal Policy Optimization Algorithms for Super Mario 
        Bros""")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="simple")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=1.0, help='parameter for GAE')
    parser.add_argument('--alpha', type=float, default=0.5, help='value loss')
    parser.add_argument('--beta', type=float, default=0.01, help='entropy coefficient')
    parser.add_argument('--epsilon_clip', type=float, default=0.2, help='parameter for Clipped Surrogate Objective')
    parser.add_argument('--update_steps', type=int, default=500, help='target_net update steps')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--k_epochs', type=int, default=3)
    parser.add_argument('--max_episodes', type=int, default=1000000, help='max_episodes')
    parser.add_argument("--save_interval", type=int, default=50, help="Number of steps between savings")
    parser.add_argument("--max_actions", type=int, default=200, help="Maximum repetition steps in test phase")
    parser.add_argument("--log_path", type=str, default="tensorboard/ppo_super_mario_bros_simple-1")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument('--render', type=bool, default=False)
    args = parser.parse_args()
    return args



class Memory:
    # action 行为，states 状态，logprobs概率 rewards奖励,is_terminals是否终结
    def __init__(self):
        self.actions = []
        self.rewards = []
        self.states = []
        self.logprobs = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.rewards[:]
        del self.states[:]
        del self.logprobs[:]
        del self.is_terminals[:]


class PPO:
    def __init__(self, opt, num_inputs, num_actions):
        self.loss_count = 0
        self.lr = opt.lr
        self.gamma = opt.gamma
        self.K_epochs = opt.update_steps
        self.eps_clip = opt.epsilon_clip
        self.tau = opt.tau
        self.beta = opt.beta
        self.alpha = opt.alpha
        self.k_epochs = opt.k_epochs
        self.policy = Net(num_inputs, num_actions).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.policy_old = Net(num_inputs, num_actions).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss().to(device)

    def update(self, memory, writer):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            # 每一步得分衰减
            discounted_reward = discounted_reward * self.gamma + reward
            # 插入每一步得分
            rewards.insert(0, discounted_reward)
        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        # Optimize policy for K epochs:
        for _ in range(self.k_epochs):
            self.loss_count += 1
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            # Finding the ratio (pi_theta / pi_theta__old):
            ratio = torch.exp(logprobs - old_logprobs.detach())
            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            tmp1 = ratio * advantages
            tmp2 = torch.clip(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            losses = -torch.min(tmp1, tmp2) + self.alpha * self.MseLoss(state_values,
                                                                        rewards) - self.beta * dist_entropy
            # take gradient step
            self.optimizer.zero_grad()
            losses = losses.mean()
            losses.backward()
            self.optimizer.step()
            writer.add_scalar("loss", losses, self.loss_count)
        # Copy new weights into old policy:
        # self.policy_old.load_state_dict(self.policy.state_dict())
        for param, target_param in zip(self.policy.parameters(), self.policy_old.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


def eval_game(opt, model_path):
    env, num_states, num_actions = create_env(opt.world, opt.stage, opt.action_type)
    ppo = PPO(opt, num_states, num_actions)
    ppo.policy.load_state_dict(torch.load(model_path), strict=False)
    state = env.reset()
    x_pos = 0
    while True:
        state = np.array(state, dtype=np.float32)
        state = torch.tensor(state).to(device)
        x = ppo.policy.common_layer(state)
        action = ppo.policy.actor_layer(x.view(-1, 3136)).argmax().item()
        state_, reward, done, info = env.step(action)
        if opt.render:
            env.render()
        if done:
            if info['flag_get']:
                return info['x_pos'], True
            x_pos = info['x_pos']
            break
        state = state_
    return x_pos, False


def train(opt):
    # torch.manual_seed(123)
    writer = SummaryWriter(opt.log_path)
    env, num_states, num_actions = create_env(opt.world, opt.stage, opt.action_type)
    ppo = PPO(opt, num_states, num_actions)
    ppo.policy.load_state_dict(torch.load("./model/PPO_" + "Mario_v0_{}_{}_best_x_pos".format(opt.world,
                                                                                              opt.stage) + ".pth"),
                               strict=False)
    memory = Memory()
    sum_steps = 0
    sum_reward = 0
    sum_x_pos = 0
    eval_during = 100
    during = 10
    best_x_pos = 0
    best_x_pos_idx = 0
    for i_episode in range(1, opt.max_episodes + 1):
        state = env.reset()
        while True:
            action = ppo.policy.act(state, memory)
            state_, reward, done, info = env.step(action)
            sum_reward += reward
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            sum_steps += 1
            if sum_steps % opt.update_steps == 0:
                ppo.update(memory, writer)
                memory.clear_memory()
            if opt.render:
                env.render()
            if done:
                sum_x_pos += info['x_pos']
                break
            state = state_
        if i_episode % during == 0:
            writer.add_scalar("rewards/i_episode_{}".format(during), sum_reward / during, i_episode)
            writer.add_scalar("x_pos/i_episode_{}".format(during), sum_x_pos / during, i_episode)
            sum_reward = 0
            sum_x_pos = 0
        if i_episode % eval_during == 0:
            model_path = "./model/Every_100_Episode_PPO_" + "Mario_v0_{}_{}".format(opt.world,
                                                                                    opt.stage) + ".pth"
            torch.save(ppo.policy.state_dict(), model_path)

            x_pos, flag = eval_game(opt, model_path)
            writer.add_scalar("X_pos_eval", x_pos, i_episode // eval_during)
            if x_pos > best_x_pos:
                best_x_pos = x_pos
                best_x_pos_idx += 1
                writer.add_scalar('best_x_pos/Every_100_Episode', best_x_pos, best_x_pos_idx)
                model_path = "./model/PPO_" + "Mario_v0_{}_{}_best_x_pos".format(opt.world,
                                                                                 opt.stage) + ".pth"
                torch.save(ppo.policy.state_dict(), model_path)
            if flag:
                model_path = "./model/PPO_" + "Mario_v0_{}_{}".format(opt.world,
                                                                      opt.stage) + ".pth"
                torch.save(ppo.policy.state_dict(), model_path)


if __name__ == '__main__':
    args = get_args()
    train(args)
