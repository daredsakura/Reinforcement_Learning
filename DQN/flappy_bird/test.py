import os
import argparse
import time

import gym_super_mario_bros
import torch
import numpy as np
from ple import PLE
from ple.games import FlappyBird
from src.env import create_env
import torch.nn.functional as F
from train import DQN

device = torch.device('cuda:0')


# 对世界关卡和训练超参数通过命令行设置
def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Asynchronous Methods for Deep Reinforcement Learning for 
        flappy bird""")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.95, help='discount factor for rewards')
    parser.add_argument('--epsilon', type=float, default=1, help='exploration rate start')
    parser.add_argument('--alpha', type=float, default=0.6, help='alpha for PER')
    parser.add_argument('--beta_init', type=float, default=0.4, help='beta for PER')
    parser.add_argument('--beta_gain_steps', type=int, default=int(3e5), help='steps of beta from beta_init to 1.0')
    parser.add_argument('--max_episodes', type=int, default=1000000, help='max_episodes')
    parser.add_argument('--target_update_steps', type=int, default=200, help='target_net update steps')
    parser.add_argument('--memory_capacity', type=int, default=200000, help='ReplayBuffer size')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size of ReplayBuffer')
    parser.add_argument("--log_path", type=str, default="tensorboard/dqn_flappy_bird")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--load_from_previous_stage", type=bool, default=False,
                        help="Load weight from previous trained sta ge")
    parser.add_argument("--use_gpu", type=bool, default=True)

    args = parser.parse_args()
    return args


def test(opt):
    # writer = SummaryWriter(opt.log_path)
    env = FlappyBird()
    reward_values = {
        "positive": 1,
        "tick": 0.1,
        "loss": -1,
    }
    p = PLE(env, fps=30, display_screen=True, force_fps=False, reward_values=reward_values)
    num_actions = len(p.getActionSet())
    num_states = len(p.getGameState())
    dqn = DQN(opt.gamma, opt.lr, opt.epsilon, num_actions, num_states, opt.batch_size, opt.target_update_steps)
    model_path = "./trained_models/Double_Dueling_DQN_Flappy_bird.pth"
    dqn.eval_net.load_state_dict(torch.load(model_path), strict=False)
    dqn.eval_net.eval()

    for _ in range(10):
        p.reset_game()
        state = np.array(list(p.getGameState().values()), dtype=np.float32)
        step = 0
        while True:
            state = torch.tensor(state).clone().detach().to(device)
            action_idx = dqn.eval_net(state).argmax().item()
            action = p.getActionSet()[action_idx]
            reward, state_, done = p.act(action), np.array(list(p.getGameState().values()),
                                                           dtype=np.float32), p.game_over()
            if done:
                break
            step += 1
            state = state_
        print(step)


if __name__ == "__main__":
    opt = get_args()
    test(opt)
