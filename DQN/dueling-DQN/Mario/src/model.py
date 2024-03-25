import torch
import torch.nn.functional as F
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)  # 84*84*3 ->42*42*32
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)  # 42*42*32 -> 21*21*32
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)  # 21*21*32 -> 11*11*32
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)  # 11*11*32 -> 6*6*32
        # self.fla = nn.Flatten()
        self.fc1 = nn.Linear(1152, 512)
        self.critic_layer = nn.Linear(512, 1)
        self.actor_layer = nn.Linear(512, num_actions)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 32 * 6 * 6)
        x = F.relu(self.fc1(x))
        A = self.actor_layer(x)
        V = self.critic_layer(x)
        return A + V - A.mean(0)
