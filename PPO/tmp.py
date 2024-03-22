# gym安装：pip install gym matplotlib -i  https://pypi.tuna.tsinghua.edu.cn/simple
import random
import torch
import torch.nn as nn
import numpy as np
import gym


# from torch.utils.tensorboard import SummaryWriter


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, 3)
        )
        self.MSELoss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, inputs):
        return self.fc(inputs)


env = gym.make('MountainCar-v0')
env = env.unwrapped
DQN = Network()  # DQN network, 需要训练的网络
Target_net = Network()  # Target network

# writer = SummaryWriter("logs_DQN_MountainCar")   # 注意tensorboard的部分


stored_transition_cnt = 0  # 记录transition_cnt的次数
replay_buffer_size = 2000  # buffer size
discount_factor = 0.6  # 衰减系数
transition_cnt = 0  # 记录发生的transition的总次数
update_interval = 20  # 将net的参数load到net2的间隔
gamma = 0.9  # 折扣因子
batch_size = 1000  # batch size
replay_buffer = np.zeros((replay_buffer_size, 6))  # 初始化buffer 列中储存 s, a, state_, r
start_learning = False  # 标记是否开始学习
Max_epoch = 50000  # 学习的回合数
epsilon = 0.1

graph_added = False
for i in range(Max_epoch):
    state = env.reset()  # 重置环境
    step = 0
    while True:
        if random.randint(0, 100) < 100 * (discount_factor ** transition_cnt):  # act greedy, 就是随机探索，刚开始所及探索多，后面变少
            action = random.randint(0, 2)
        else:
            output = DQN(torch.Tensor(state)).detach()  # output中是[左走累计奖励, 右走累计奖励]
            action = torch.argmax(output).data.item()  # 用argmax选取动作
        state_, reward, done, _ = env.step(action)  # 执行动作，获得env的反馈
        # 自己定义一个reward
        # 只根据小车的位置给reward
        # reward = state_[0] + 0.5
        if state_[0] <= -0.5:
            reward = 100 * abs(state_[1])
            # print('速度：', state_[1])
        elif -0.5 < state_[0] < 0.5:
            reward = pow(2, 5 * (state_[0] + 1)) + (100 * abs(state_[1])) ** 2
        elif state_[0] >= 0.5:
            reward = 1000

        replay_buffer[stored_transition_cnt % replay_buffer_size][0:2] = state
        replay_buffer[stored_transition_cnt % replay_buffer_size][2:3] = action
        replay_buffer[stored_transition_cnt % replay_buffer_size][3:5] = state_
        replay_buffer[stored_transition_cnt % replay_buffer_size][5:6] = reward
        stored_transition_cnt += 1
        state = state_
        step += 1
        if stored_transition_cnt > replay_buffer_size:
            # 如果到达update_interval，则将net的参数load到net2中
            if transition_cnt % update_interval == 0:
                Target_net.load_state_dict(DQN.state_dict())
                step /= 20
                print("i={},step={}".format(i, step))
                step = 0
            # 从replay buffer中提取一个batch，注意可以是随机提取.
            # 提取之后将其转成tensor数据类型，以便输入给神经网络
            index = random.randint(0, replay_buffer_size - batch_size - 1)
            batch_state = torch.Tensor(replay_buffer[index:index + batch_size, 0:2])
            batch_action = torch.Tensor(replay_buffer[index:index + batch_size, 2:3]).long()
            batch_state_ = torch.Tensor(replay_buffer[index:index + batch_size, 3:5])
            batch_reward = torch.Tensor(replay_buffer[index:index + batch_size, 5:6])

            # 用tensorboard可视化神经网络
            # if(graph_added == False):
            #     writer.add_graph(model=DQN, input_to_model=batch_state)
            #     writer.add_graph(model=Target_net, input_to_model=batch_state)
            #     graph_added = True

            # 训练-更新网络：gradient descent updates
            # 我们用Target_net来计算TD-target
            q = DQN(batch_state).gather(1, batch_action)  # predict q-value by old network
            q_next = Target_net(batch_state_).detach().max(1)[0].reshape(batch_size, 1)  # predict q(s_t+1)
            q_target = batch_reward + gamma * q_next  # 用Target_net来计算TD-target
            loss = DQN.MSELoss(q, q_target)  # 计算loss
            DQN.optimizer.zero_grad()  # 将DQN上步的梯度清零
            loss.backward()  # DQN反向传播，更新参数
            DQN.optimizer.step()  # DQN更新参数

            transition_cnt += 1
            if not start_learning:
                print('start learning')
                start_learning = True
                break
        if done:
            break

        env.render()

torch.save(DQN.state_dict(), 'DQN_MountainCar-v0.pth')

# writer.close()
