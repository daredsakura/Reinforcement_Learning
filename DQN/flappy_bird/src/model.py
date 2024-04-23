import torch
import torch.nn.functional as F
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(Net, self).__init__()
        self.common_layer = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.critic_layer = nn.Sequential(
            nn.Linear(128, 1)
        )
        self.actor_layer = nn.Sequential(
            nn.Linear(128, num_actions)
        )
        # self.common_layer = nn.Sequential(
        #     # conv1
        #     nn.Conv2d(in_channels=4,
        #               out_channels=32,
        #               kernel_size=8,
        #               stride=4),
        #     nn.ReLU(),
        #
        #     # conv2
        #     nn.Conv2d(in_channels=32,
        #               out_channels=64,
        #               kernel_size=4,
        #               stride=2),
        #     nn.ReLU(),
        #
        #     # conv3
        #     nn.Conv2d(in_channels=64,
        #               out_channels=64,
        #               kernel_size=3,
        #               stride=1),
        #     nn.ReLU()
        # )
        # self.critic_layer = nn.Sequential(
        #     nn.Linear(in_features=64 * 7 * 7, out_features=512),
        #     nn.ReLU(),
        #     nn.Linear(in_features=512, out_features=1))
        # self.actor_layer = nn.Sequential(
        #     nn.Linear(in_features=64 * 7 * 7, out_features=512),
        #     nn.ReLU(),
        #     nn.Linear(in_features=512, out_features=num_actions)
        # )
        # self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.common_layer(x)
        # x = x.view(-1, 64 * 7 * 7)
        A = self.actor_layer(x)
        V = self.critic_layer(x)
        return A + V - A.mean(0)
