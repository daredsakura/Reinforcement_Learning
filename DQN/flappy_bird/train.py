import argparse
import torch
from ple import PLE
from ple.games import FlappyBird
from src.model import Net
import random
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import numpy as np

device = torch.device('cuda:0')


# 对世界关卡和训练超参数通过命令行设置
def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Asynchronous Methods for Deep Reinforcement Learning for 
        flappy bird""")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.95, help='discount factor for rewards')
    parser.add_argument('--epsilon', type=float, default=0.3, help='exploration rate start')
    parser.add_argument('--alpha', type=float, default=0.6, help='alpha for PER')
    parser.add_argument('--beta_init', type=float, default=0.4, help='beta for PER')
    parser.add_argument('--beta_gain_steps', type=int, default=int(3e5), help='steps of beta from beta_init to 1.0')
    parser.add_argument('--max_episodes', type=int, default=1000000, help='max_episodes')
    parser.add_argument('--target_update_steps', type=int, default=200, help='target_net update steps')
    parser.add_argument('--memory_capacity', type=int, default=200000, help='ReplayBuffer size')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size of ReplayBuffer')
    parser.add_argument("--log_path", type=str, default="tensorboard/dqn_flappy_bird-1")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--load_from_previous_stage", type=bool, default=False,
                        help="Load weight from previous trained sta ge")
    parser.add_argument("--use_gpu", type=bool, default=True)

    args = parser.parse_args()
    return args


# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree(object):
    # """
    # Story data with its priority in the tree.
    # Tree structure and array storage:
    #
    # Tree index:
    #      0         -> storing priority sum
    #     / \
    #   1     2
    #  / \   / \
    # 3   4 5   6    -> storing priority for transitions
    #
    # Array type for storing:
    # [0,1,2,3,4,5,6]
    # """

    def __init__(self, buffer_capacity):
        self.buffer_capacity = buffer_capacity  # buffer的容量
        self.tree_capacity = 2 * buffer_capacity - 1  # sum_tree的容量
        self.tree = np.zeros(self.tree_capacity)

    def update_priority(self, data_index, priority):
        """ Update the priority for one transition according to its index in buffer """
        # data_index表示当前数据在buffer中的index
        # tree_index表示当前数据在sum_tree中的index
        tree_index = data_index + self.buffer_capacity - 1  # 把当前数据在buffer中的index转换为在sum_tree中的index
        change = priority - self.tree[tree_index]  # 当前数据的priority的改变量
        self.tree[tree_index] = priority  # 更新树的最后一层叶子节点的优先级
        # then propagate the change through the tree
        while tree_index != 0:  # 更新上层节点的优先级，一直传播到最顶端
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    def prioritized_sample(self, N, batch_size, beta):
        """ sample a batch of index and normlized IS weight according to priorites """
        batch_index = np.zeros(batch_size, dtype=np.uint32)
        IS_weight = torch.zeros(batch_size, dtype=torch.float32)
        segment = self.priority_sum / batch_size  # 把[0,priority_sum]等分成batch_size个区间，在每个区间均匀采样一个数
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            v = np.random.uniform(a, b)
            buffer_index, priority = self._get_index(v)
            batch_index[i] = buffer_index
            prob = priority / self.priority_sum  # 当前数据被采样的概率
            IS_weight[i] = (N * prob) ** (-beta)
        Normed_IS_weight = IS_weight / IS_weight.max()  # normalization

        return batch_index, Normed_IS_weight

    def _get_index(self, v):
        """ sample a index """
        parent_idx = 0  # 从树的顶端开始
        while True:
            child_left_idx = 2 * parent_idx + 1  # 父节点下方的左右两个子节点的index
            child_right_idx = child_left_idx + 1
            if child_left_idx >= self.tree_capacity:  # reach bottom, end search
                tree_index = parent_idx  # tree_index表示采样到的数据在sum_tree中的index
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[child_left_idx]:
                    parent_idx = child_left_idx
                else:
                    v -= self.tree[child_left_idx]
                    parent_idx = child_right_idx

        data_index = tree_index - self.buffer_capacity + 1  # tree_index->data_index
        return data_index, self.tree[tree_index]  # 返回采样到的data在buffer中的index,以及相对应的priority

    @property
    def priority_sum(self):
        return self.tree[0]  # 树的顶端保存了所有priority之和

    @property
    def priority_max(self):
        return self.tree[self.buffer_capacity - 1:].max()  # 树的最后一层叶节点，保存的才是每个数据对应的priority


class PrioritizedBuffer(object):
    def __init__(self, opt, num_states):
        self.ptr = 0
        self.size = 0

        max_size = int(opt.memory_capacity)
        self.state = np.zeros((max_size, num_states))
        self.action = np.zeros((max_size, 1))
        self.reward = np.zeros((max_size, 1))
        self.next_state = np.zeros((max_size, num_states))
        self.dw = np.zeros((max_size, 1))
        self.max_size = max_size

        self.sum_tree = SumTree(max_size)
        self.alpha = opt.alpha
        self.beta = opt.beta_init

        self.device = device

    def add(self, state, action, reward, next_state, dw):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.dw[self.ptr] = dw  # 0,0,0，...，1

        # 如果是第一条经验，初始化优先级为1.0；否则，对于新存入的经验，指定为当前最大的优先级
        priority = 1.0 if self.size == 0 else self.sum_tree.priority_max
        self.sum_tree.update_priority(data_index=self.ptr, priority=priority)  # 更新当前经验在sum_tree中的优先级

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind, Normed_IS_weight = self.sum_tree.prioritized_sample(N=self.size, batch_size=batch_size, beta=self.beta)

        return (
            torch.tensor(self.state[ind], dtype=torch.float32).to(self.device),
            torch.tensor(self.action[ind], dtype=torch.long).to(self.device),
            torch.tensor(self.reward[ind], dtype=torch.float32).to(self.device),
            torch.tensor(self.next_state[ind], dtype=torch.float32).to(self.device),
            torch.tensor(self.dw[ind], dtype=torch.float32).to(self.device),
            ind,
            Normed_IS_weight.to(self.device)  # shape：(batch_size,)
        )

    def update_batch_priorities(self, batch_index, td_errors):  # 根据传入的td_error，更新batch_index所对应数据的priorities
        priorities = (np.abs(td_errors) + 0.01) ** self.alpha
        for index, priority in zip(batch_index, priorities):
            self.sum_tree.update_priority(data_index=index, priority=priority)


#
# class ReplayBuffer:
#     def __init__(self, MEMORY_CAPACITY):
#         self.buffer = deque(maxlen=MEMORY_CAPACITY)
#
#     def add(self, state, action, reward, state_, done):
#         self.buffer.append((state, action, reward, state_, done))
#
#     def __len__(self):
#         return len(self.buffer)
#
#     def sample(self, batch_size):
#         sub_buffer = random.sample(self.buffer, batch_size)
#         state, action, reward, state_, done = zip(*sub_buffer)  # *代表解压
#         return np.array(state), action, reward, np.array(state_), done


class DQN:
    def __init__(self, GAMMA, LR, EPSILON, num_actions, num_states, batch_size, target_update_steps):
        self.eval_net, self.target_net = Net(num_states, num_actions).to(device), Net(num_states,
                                                                                      num_actions,
                                                                                      ).to(
            device)
        self.learn_count = 0
        self.curr_step = 0
        self.lr = LR
        self.epsilon = EPSILON
        self.epsilon_final = 0.0001
        self.epsilon_decay = 1e-7
        self.learn_every = 5
        self.frame_per_action = 1
        self.gamma = GAMMA
        self.tau = 0.005  # 控制目标网络参数和目标值网络参数之间的更新速度 ,目标网络参数 = (1 - tau) * 目标网络参数 + tau * 主要网络参数
        self.action_dim = num_actions
        self.batch_size = batch_size
        self.target_update_step = target_update_steps
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(),
                                          lr=self.lr,
                                          )
        self.loss_fn = torch.nn.MSELoss()

    def choose_action(self, state):
        state = np.array(state, dtype=np.float32)
        state = torch.tensor(state).to(device)
        if self.curr_step % self.frame_per_action and self.curr_step <= 10000:
            action = 1
        else:
            if np.random.random() < self.epsilon:
                action = np.random.randint(self.action_dim)
            else:
                actions = self.eval_net(state)
                action = actions.argmax().item()

        return action

    def learn(self, replay_buffer):
        if self.learn_count % self.target_update_step == 0:
            for param, target_param in zip(self.eval_net.parameters(), self.target_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        state, action, reward, state_, done, indices, weights = replay_buffer.sample(self.batch_size)
        # state = torch.FloatTensor(state).to(device)
        # action = torch.tensor(action).view(-1, 1).to(device)
        # reward = torch.tensor(reward, dtype=torch.float).view(-1, 1).to(device)
        # done = torch.tensor(done, dtype=torch.float).view(-1, 1).to(device)
        # state_ = torch.FloatTensor(state_).to(device)
        # Double-DQN
        # q_target=reward+gamma*Q(s_,argmax_(a_)Q(s_,a_,params_eval),params_target)
        with torch.no_grad():
            action_ = self.eval_net(state_).argmax(dim=1).unsqueeze(-1)
            max_next_q_value = self.target_net(state_).gather(1, action_)
            q_target = reward + self.gamma * max_next_q_value * (1 - done)
        q_value = self.eval_net(state).gather(1, action)
        td_errors = (q_value - q_target).squeeze(-1)
        losses = (weights * (td_errors ** 2)).mean()
        # losses = losses.mean()
        self.optimizer.zero_grad()
        losses.backward()
        # torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), 10.0)
        for param in self.eval_net.parameters():  # clip防止梯度爆炸
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.learn_count += 1
        replay_buffer.update_batch_priorities(indices, td_errors.detach().cpu().numpy())
        return losses


def eval_model(model_path, opt, p):
    num_actions = len(p.getActionSet())
    num_states = len(p.getGameState())
    dqn = DQN(opt.gamma, opt.lr, opt.epsilon, num_actions, num_states, opt.batch_size, opt.target_update_steps)
    dqn.eval_net.load_state_dict(torch.load(model_path), strict=False)
    dqn.eval_net.eval()
    step = 0
    for _ in range(10):
        p.reset_game()
        state = np.array(list(p.getGameState().values()), dtype=np.float32)
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
    return step // 10


def train(opt):
    # torch.manual_seed(123)
    writer = SummaryWriter(opt.log_path)
    env = FlappyBird()
    reward_values = {
        "positive": 1,
        "tick": 0.1,
        "loss": -1,
    }
    p = PLE(env, fps=30, display_screen=False, reward_values=reward_values)
    p_test = PLE(env, fps=30, display_screen=False, reward_values=reward_values)
    # print(p.getActionSet())
    num_actions = len(p.getActionSet())
    num_states = len(p.getGameState())
    dqn = DQN(opt.gamma, opt.lr, opt.epsilon, num_actions, num_states, opt.batch_size, opt.target_update_steps)
    model_path = "./trained_models/Double_Dueling_DQN_Flappy_bird.pth"
    dqn.eval_net.load_state_dict(torch.load(model_path), strict=False)
    dqn.target_net.load_state_dict(dqn.eval_net.state_dict())
    dqn.target_net.eval()
    replay_buffer = PrioritizedBuffer(opt, num_states)
    update_index = 0
    sum_reward = 0
    sum_step = 0
    sum_loss = 0
    during = 10
    best_step = 0
    best_step_idx = 0
    for i_episode in range(1, opt.max_episodes, 1):
        p.reset_game()
        state = list(p.getGameState().values())
        while True:
            action_idx = dqn.choose_action(state)
            action = p.getActionSet()[action_idx]
            # print(action)
            reward, state_, done = p.act(action), list(p.getGameState().values()), p.game_over()
            sum_reward += reward
            replay_buffer.add(state, action_idx, reward, state_, done)
            if dqn.curr_step >= 10000 and dqn.curr_step % dqn.learn_every == 0:
                sum_loss += dqn.learn(replay_buffer).item()
                dqn.epsilon = max(dqn.epsilon - dqn.epsilon_decay, dqn.epsilon_final)
            if done:
                break
            dqn.curr_step += 1
            update_index += 1
            sum_step += 1
            state = state_
        if i_episode % during == 0:
            writer.add_scalar("rewards/i_episode_{}".format(during), sum_reward / during, i_episode)
            writer.add_scalar("sum_loss/i_episode_{}".format(during), sum_loss / during, i_episode)
            writer.add_scalar("sum_step/i_episode_{}".format(during), sum_step / during, i_episode)
            sum_reward = 0
            sum_loss = 0
            sum_step = 0
        if i_episode % 100 == 0:
            model_path = "./trained_models/Every_100_Episode_Double_Dueling_DQN_Flappy_bird.pth"
            torch.save(dqn.eval_net.state_dict(), model_path)
            step = eval_model(model_path, opt, p_test)
            writer.add_scalar('eval_step/i_episode_100', step, i_episode // 100)
            if step > best_step:
                best_step_idx += 1
                best_step = max(step, best_step)
                model_path = "./trained_models/Double_Dueling_DQN_Flappy_bird.pth"
                torch.save(dqn.eval_net.state_dict(), model_path)
                writer.add_scalar('eval_best_step/i_episode', best_step, best_step_idx)

    writer.close()


if __name__ == '__main__':
    args = get_args()
    train(args)
