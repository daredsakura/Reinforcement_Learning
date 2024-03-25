import torch
from src.env import create_train_env
from src.model import ActorCritic
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import timeit  # 用于测量代码的执行时间。


def train(index, opt, global_model, optimizer, save=False):  # index :进程编号
    torch.manual_seed(123 + index)
    if save:
        start_time = timeit.default_timer()
    writer = SummaryWriter(opt.log_path)
    env, num_states, num_actions = create_train_env(opt.world, opt.stage, opt.action_type)
    local_model = ActorCritic(num_states, num_actions)
    device = torch.device("cuda:0" if opt.use_gpu else "cpu")
    local_model.to(device)
    local_model.train()
    state = torch.from_numpy(env.reset()).to(device)
    cur_episode = 0
    cur_step = 0
    done = True
    h_0 = torch.zeros((1, 512), dtype=torch.float).to(device)
    c_0 = torch.zeros((1, 512), dtype=torch.float).to(device)
    reward_e_game = 0
    cnt_game = 0
    while True:
        if save:
            if cur_episode % opt.save_interval == 0 and cur_episode > 0:
                torch.save(global_model.state_dict(),
                           "{}/a3c_super_mario_bros_{}_{}".format(opt.saved_path, opt.world, opt.stage))
            # print("Process {}. Episode {}:".format(index, cur_episode))
        cur_episode += 1
        local_model.load_state_dict(global_model.state_dict())
        if done:
            h_0 = torch.zeros((1, 512), dtype=torch.float).to(device)
            c_0 = torch.zeros((1, 512), dtype=torch.float).to(device)
        else:
            h_0 = h_0.detach().to(device)
            c_0 = c_0.detach().to(device)

        log_policies = []
        values = []
        rewards = []
        entropies = []
        for _ in range(opt.num_local_steps):
            cur_step += 1
            actions, value, h_0, c_0 = local_model(state, h_0, c_0)
            policy = F.softmax(actions, dim=1)
            log_policy = F.log_softmax(actions, dim=1)
            # entropy = -torch.mul(policy, log_policy.T).view(-1, 1)
            entropy = -(policy * log_policy).sum(1, keepdim=True)
            dist = Categorical(policy)
            action = dist.sample().item()
            state, reward, done, info = env.step(action)
            state = torch.from_numpy(state).to(device)
            if cur_step > opt.num_local_steps:
                done = True
            if done:
                reward_e_game += reward
                cnt_game += 1
                writer.add_scalar("Train_{}_game_value".format(index), reward_e_game, cnt_game)
                cur_step = 0
                reward_e_game = 0
                state = torch.from_numpy(env.reset()).to(device)
            values.append(value)
            log_policies.append(log_policy[0, action])
            rewards.append(reward)
            entropies.append(entropy)
            if done:
                break
        R = torch.zeros((1, 1), dtype=torch.float).to(device)
        if not done:
            _, R, _, _ = local_model(state, h_0, c_0)
        gae = torch.zeros((1, 1), dtype=torch.float).to(device)  # 广义优势估计，代替actor_loss中的A(优势函数)
        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        next_value = R
        for value, reward, log_policy, entropy in list(zip(values, rewards, log_policies, entropies))[::-1]:
            gae = gae * opt.gamma * opt.tau
            gae = gae + reward + opt.gamma * next_value.detach() - value.detach()
            next_value = value
            actor_loss = actor_loss + log_policy * gae
            R = R * opt.gamma + reward
            critic_loss = critic_loss + (R - value) ** 2 / 2
            entropy_loss = entropy_loss + entropy
        total_loss = -actor_loss + critic_loss - opt.beta * entropy_loss
        writer.add_scalar("Train_{}".format(index), total_loss, cur_episode)
        optimizer.zero_grad()
        total_loss.backward()
        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            if global_param.grad is not None:
                break
            global_param._grad = local_param.grad
        optimizer.step()
        if cur_episode == opt.num_global_steps // opt.num_local_steps:
            print("Training process {} terminated".format(index))
            if save:
                end_time = timeit.default_timer()
                print('The code runs for %.2f s .' % (end_time - start_time))
            return
