# import os
import argparse
import multiprocessing as mp
import torch
from src.env import create_env, create_train_env
from src.model import Net
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
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
    parser.add_argument('--num_envs', type=int, default=8, help='num of processes')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--k_epochs', type=int, default=3)
    parser.add_argument('--num_local_steps', type=int, default=512)
    parser.add_argument('--max_episodes', type=int, default=1000000, help='max_episodes')
    parser.add_argument("--save_interval", type=int, default=50, help="Number of steps between savings")
    parser.add_argument("--max_actions", type=int, default=200, help="Maximum repetition steps in test phase")
    parser.add_argument("--log_path", type=str, default="/tf_logs")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument('--render', type=bool, default=False)
    args = parser.parse_args()
    return args


class MultipleEnvironments:
    def __init__(self, opt, output_path=None):
        # 创建多个pipe 通信管道,用于实现多进程环境的样本采集。
        self.agent_conns, self.env_conns = zip(*[mp.Pipe() for _ in range(opt.num_envs)])
        if opt.action_type == "right":
            actions = RIGHT_ONLY
        elif opt.action_type == "simple":
            actions = SIMPLE_MOVEMENT
        else:
            actions = COMPLEX_MOVEMENT
        # 创建多环境
        self.envs = [create_train_env(opt.world, opt.stage, opt.action_type, output_path=output_path) for _ in
                     range(opt.num_envs)]
        self.num_states = self.envs[0].observation_space.shape[0]
        self.num_actions = len(actions)
        #  创建多进程
        for index in range(opt.num_envs):
            process = mp.Process(target=self.run, args=(index,))
            process.start()
            self.env_conns[index].close()

    def run(self, index):
        self.agent_conns[index].close()
        while True:
            request, action = self.env_conns[index].recv()
            if request == 'step':
                self.env_conns[index].send(self.envs[index].step(int(action)))
            elif request == 'reset':
                self.env_conns[index].send(self.envs[index].reset())
            else:
                raise NotImplementedError


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
        self.memory = Memory()

    def update(self, writer, opt):
        all_states = np.array(self.memory.states)
        all_actions = np.array(self.memory.actions)
        all_logprobs = np.array(self.memory.logprobs)
        # print(logprobs.shape)
        for i in range(opt.num_envs):
            # Monte Carlo estimate of state rewards:
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
                if is_terminal[i]:
                    discounted_reward = 0
                # 每一步得分衰减
                discounted_reward = discounted_reward * self.gamma + reward[i]
                # 插入每一步得分
                rewards.insert(0, discounted_reward)
            # Normalizing the rewards:
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            # convert ndarray to tensor
            old_states = torch.tensor(all_states[:, i, :, :]).to(device).detach()
            old_actions = torch.tensor(all_actions[:, i]).to(device).detach()
            old_logprobs = torch.tensor(all_logprobs[:, i]).to(device).detach()
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

    if opt.action_type == "right":
        num_actions = len(RIGHT_ONLY)
    elif opt.action_type == "simple":
        num_actions = len(SIMPLE_MOVEMENT)
    else:
        num_actions = len(COMPLEX_MOVEMENT)
    envs = MultipleEnvironments(opt)
    num_states = envs.envs[0].observation_space.shape[0]
    ppo = PPO(opt, num_states, num_actions)
    sum_steps = 0
    sum_reward = 0
    sum_x_pos = 0
    eval_during = 5
    during = 10
    best_x_pos = 0
    best_x_pos_idx = 0
    [agent_conn.send(('reset', None)) for agent_conn in envs.agent_conns]
    states = [agent_conn.recv() for agent_conn in envs.agent_conns]
    for i_episode in range(1, opt.max_episodes + 1):
        for _ in range(opt.num_local_steps):
            actions = ppo.policy.act(states, ppo.memory)
            # print(actions)
            [agent_conn.send(("step", act)) for agent_conn, act in
             zip(envs.agent_conns, actions.cpu().numpy().astype("int16").tolist())]
            states_, rewards, dones, infos = zip(*[agent_conn.recv() for agent_conn in envs.agent_conns])
            states_ = list(states_)
            rewards = list(rewards)
            dones = list(dones)
            # print(dones)
            for i in range(len(dones)):
                if dones[i]:
                    envs.agent_conns[i].send(('reset', None))
                    states_[i] = envs.agent_conns[i].recv()
            sum_reward += np.mean(rewards)
            ppo.memory.rewards.append(rewards)
            ppo.memory.is_terminals.append(dones)
            sum_steps += 1
            states = states_
        ppo.update(writer, opt)
        ppo.memory.clear_memory()
        writer.add_scalar("rewards/i_episode_{}".format(during), sum_reward , i_episode)
        writer.add_scalar("x_pos/i_episode_{}".format(during), sum_x_pos, i_episode)
        sum_reward = 0
        sum_x_pos = 0
        if i_episode % eval_during == 0:
            model_path = "./model/Every_5_Episode_PPO_" + "Mario_v0_{}_{}".format(opt.world,
                                                                                   opt.stage) + ".pth"
            torch.save(ppo.policy.state_dict(), model_path)
            model_path = "./model/Every_5_Episode_PPO_" + "Mario_v0_{}_{}".format(opt.world,
                                                                                   opt.stage) + ".pth"
            x_pos, flag = eval_game(opt, model_path)
            writer.add_scalar("X_pos_eval", x_pos, i_episode // eval_during)
            if x_pos > best_x_pos:
                best_x_pos = x_pos
                best_x_pos_idx += 1
                writer.add_scalar('best_x_pos/Every_5_Episode', best_x_pos, best_x_pos_idx)
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
