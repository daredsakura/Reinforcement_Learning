import os
import argparse
import time

import gym_super_mario_bros
import torch
from src.env import create_env
import torch.nn.functional as F
from train import PPO
import numpy as np

device = torch.device('cuda:0')


# 对世界关卡和训练超参数通过命令行设置
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
    parser.add_argument('--render', type=bool, default=True)
    args = parser.parse_args()
    return args


def test(opt):
    # writer = SummaryWriter(opt.log_path)
    env, num_states, num_actions = create_env(opt.world, opt.stage, opt.action_type)
    ppo = PPO(opt, num_states, num_actions)
    ppo.policy.load_state_dict(torch.load(r"model/PPO_Mario_v0_1_1_best_x_pos.pth"), strict=False)
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
            break
        state = state_
        time.sleep(0.01)


if __name__ == "__main__":
    opt = get_args()
    test(opt)
