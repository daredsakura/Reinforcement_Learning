import random
import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
import pandas as pd
import gym
# from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from train import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    env_name = 'MountainCar-v0'
    model_path = "./Dueling_DQN_" + env_name + ".pth"
    RENDER = True
    env = gym.make(env_name)
    STATE_DIM = env.observation_space.shape[0]
    ACTION_DIM = env.action_space.n
    N_EPISODE = 10
    GAMMA = 0.99  # discount factor
    LR = 0.0001  # learning rate
    EPSILON = 1
    MEMORY_CAPACITY = 100000
    BATCH_SIZE = 64
    HIDDEN_LAYERS = 256
    TARGET_UPDATE_STEPS = 100
    dqn = DQN(GAMMA, LR, EPSILON, ACTION_DIM, STATE_DIM, HIDDEN_LAYERS, BATCH_SIZE, TARGET_UPDATE_STEPS)
    dqn.eval_net.load_state_dict(torch.load(model_path), strict=False)
    for i_episode in range(N_EPISODE):
        state = env.reset()
        step = 0
        while True:
            state = torch.FloatTensor(state).to(device)
            action = dqn.eval_net(state).argmax().item()
            step += 1
            state_, reward, done, _ = env.step(action)
            if RENDER:
                env.render()
            if done:
                print("I_Episode:{} step:{}".format(i_episode, step))
                break
            state = state_
    env.close()


if __name__ == '__main__':
    main()
