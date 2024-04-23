import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

device = torch.device('cuda')


class Net(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(Net, self).__init__()
        self.common_layer = nn.Sequential(
            nn.Conv2d(num_inputs, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.actor_layer = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
            nn.Softmax(dim=-1)
        )

        self.critic_layer = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
                nn.init.constant_(module.bias, 0)

    # def forward(self, x):
    #     x = F.relu(self.conv1(x))
    #     x = F.relu(self.conv2(x))
    #     x = F.relu(self.conv3(x))
    #     x = F.relu(self.conv4(x))
    #     x = x.view(-1, 3136)
    #     return self.actor_linear(x), self.critic_linear(x)

    def act(self, state, memory):
        state = np.array(state, dtype=np.float32)
        state = torch.tensor(state).to(device)

        x = self.common_layer(state)
        x = x.view(-1, 3136)
        action_probs = self.actor_layer(x)

        dist = Categorical(action_probs)  # 按照给定的概率分布来进行采样
        action = dist.sample()
        memory.actions.append(action.cpu().numpy())
        memory.logprobs.append(dist.log_prob(action).cpu().detach().numpy())
        # print(len(dist.log_prob(action)))
        memory.states.append(state.cpu().numpy())
        return action

    def evaluate(self, state, action):
        x = self.common_layer(state)
        action_probs = self.actor_layer(x.view(-1, 3136))
        # Categorical代表随机策略
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        # cricle对state评价
        value = self.critic_layer(x.view(-1, 3136))
        return action_logprobs, torch.squeeze(value), dist_entropy
