import os
import argparse
import time

import gym_super_mario_bros
import torch
from src.env import create_env
import torch.nn.functional as F
from train import DQN


# 对世界关卡和训练超参数通过命令行设置
def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Asynchronous Methods for Deep Reinforcement Learning for 
        Super Mario Bros""")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="right")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--gamma", type=float, default=0.9, help='discount factor for rewards')
    parser.add_argument('--alpha', type=float, default=0.6, help='alpha for PER')
    parser.add_argument('--beta_init', type=float, default=0.4, help='beta for PER')
    parser.add_argument('--beta_gain_steps', type=int, default=int(3e5), help='steps of beta from beta_init to 1.0')
    parser.add_argument('--epsilon', type=float, default=1.0, help='exploration rate')
    parser.add_argument('--max_episodes', type=int, default=50000, help='max_episodes')
    parser.add_argument('--target_update_steps', type=int, default=200, help='target_net update steps')
    parser.add_argument('--memory_capacity', type=int, default=18000, help='ReplayBuffer size')
    parser.add_argument('--render', type=bool, default=True, help='is Render')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size of ReplayBuffer')
    parser.add_argument("--save_interval", type=int, default=10000, help="Number of steps between savings")
    parser.add_argument("--max_actions", type=int, default=200, help="Maximum repetition steps in test phase")
    parser.add_argument("--log_path", type=str, default="tensorboard/dqn_super_mario_bros")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--load_from_previous_stage", type=bool, default=False,
                        help="Load weight from previous trained stage")
    parser.add_argument("--use_gpu", type=bool, default=True)
    args = parser.parse_args()
    return args


def test(opt):
    # writer = SummaryWriter(opt.log_path)
    env, num_states, num_actions = create_env(opt.world, opt.stage, opt.action_type)
    dqn = DQN(opt.gamma, opt.lr, opt.epsilon, num_actions, num_states, opt.batch_size, opt.target_update_steps)
    dqn.eval_net.load_state_dict(torch.load(r".\trained_models\Double_Dueling_DQN_Mario_v0_1_1_best_x_pos.pth"),
                                 strict=False)
    device = torch.device('cuda:0')
    state = env.reset()
    step = 0
    while True:
        step += 1
        state = torch.FloatTensor(state).to(device)
        # state
        action = dqn.eval_net(state).argmax().item()
        state_, reward, done, info = env.step(action)
        print(reward)
        if opt.render:
            env.render()
        if done:
            break
        state = state_
        time.sleep(0.002)
    print(step)


if __name__ == "__main__":
    opt = get_args()
    test(opt)
