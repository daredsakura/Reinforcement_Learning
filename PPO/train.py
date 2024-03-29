import gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


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


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()
        self.actor_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim),
            nn.Softmax(dim=-1)
        )
        self.reward_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        state = state.to(device)
        # print(state)
        # print(state.shape)
        action_probs = self.actor_layer(state)

        dist = Categorical(action_probs)  # 按照给定的概率分布来进行采样
        action = dist.sample()
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))
        memory.states.append(state)
        return action.item()

    def evaluate(self, state, action):
        action_probs = self.actor_layer(state)
        # Categorical代表随机策略
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        # cricle对state评价
        value = self.reward_layer(state)
        return action_logprobs, torch.squeeze(value), dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss().to(device)

    def update(self, memory):
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
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            # Finding the ratio (pi_theta / pi_theta__old):
            ratio = torch.exp(logprobs - old_logprobs.detach())
            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            tmp1 = ratio * advantages
            tmp2 = torch.clip(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            losses = -torch.min(tmp1, tmp2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            # take gradient step
            self.optimizer.zero_grad()
            losses.mean().backward()
            self.optimizer.step()
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())


def main():
    ############## Hyperparameters ##############
    env_name = "LunarLander-v2"
    # creating environment
    env = gym.make(env_name)
    # observation为8维
    state_dim = env.observation_space.shape[0]
    # action共4个
    action_dim = 4
    render = False  # if show the game's view
    solved_reward = 230  # stop training if avg_reward > solved_reward
    log_interval = 20  # print avg reward in the interval
    max_episodes = 10000  # max training episodes
    max_timesteps = 300  # max timesteps in one episode
    n_latent_var = 64  # number of variables in hidden layer
    update_timestep = 2000  # update policy every n timesteps
    lr = 0.002  # learning rate
    betas = (0.9, 0.999)
    gamma = 0.99  # discount factor
    K_epochs = 4  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO2
    random_seed = None
    #############################################

    if random_seed is not None:
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0
    # training loop
    for i_episode in range(1, max_episodes + 1, 1):
        state = env.reset()  # 初始化（重新玩）
        for t in range(max_timesteps):
            timestep += 1
            # Running policy_old
            state = torch.tensor(state).to(device)
            action = ppo.policy_old.act(state, memory)
            state, reward, done, _ = env.step(action)  # 得到（新的状态，奖励，是否终止，额外的调试信息）
            # Saving reward and is_terminal:
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            # update  if its time
            if timestep % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0
            running_reward += reward
            if render:
                env.render()
            if done:
                break
        avg_length += t
        # stop training if avg_reward > solved_reward
        # print(log_interval * solved_reward)
        if running_reward > (log_interval * solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_{}.pth'.format(env_name))
            break
        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            running_reward = int((running_reward / log_interval))
            print('Episode {}\t avg_length={}\t running_reward={}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0
    env.close()


if __name__ == '__main__':
    main()
