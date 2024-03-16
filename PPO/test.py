import gym
import torch.nn as nn
import torch
from torch.distributions import Categorical
import Box2D
import numpy as np
from PIL import Image
from train import PPO, Memory

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def test():
    ############## Hyperparameters ##############
    env_name = "LunarLander-v2"
    # creating environment
    env = gym.make(env_name)
    # observation为8维
    state_dim = env.observation_space.shape[0]
    # action共4个
    action_dim = 4
    render = True  # if show the game's view
    solved_reward = 230  # stop training if avg_reward > solved_reward
    log_interval = 20  # print avg reward in the interval
    max_episodes = 10  # max testing episodes
    max_timesteps = 300  # max timesteps in one episode
    n_latent_var = 64  # number of variables in hidden layer
    update_timestep = 2000  # update policy every n timesteps
    lr = 0.002  # learning rate
    betas = (0.9, 0.999)
    gamma = 0.99  # discount factor
    K_epochs = 4  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO2
    save_gif = False
    random_seed = None
    #############################################
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    memory = Memory()
    file_name = './PPO_{}.pth'.format(env_name)
    print(file_name)
    ppo.policy.load_state_dict(torch.load(file_name), strict=False)
    with torch.no_grad():
        for i_episode in range(1, max_episodes + 1, 1):
            i_reward = 0
            state = env.reset()
            for t in range(max_timesteps):
                state = torch.from_numpy(np.array(state)).float().to(device)
                action = ppo.policy.act(state, memory)
                state, reward, done, _ = env.step(action)
                i_reward += reward
                if render:
                    env.render()
                if save_gif:
                    img = env.render(mode='rgb_array')
                    img = Image.fromarray(img)
                    img.save('./gif/{}.jpg'.format(t))
                if done:
                    break
            print('i_episode :{} reward={}'.format(i_episode, i_reward))


if __name__ == '__main__':
    test()
