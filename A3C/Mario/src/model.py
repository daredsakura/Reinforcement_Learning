import torch
import torch.nn.functional as F
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)  # 84*84*3 ->42*42*32
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)  # 42*42*32 -> 21*21*32
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)  # 21*21*32 -> 11*11*32
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)  # 11*11*32 -> 6*6*32
        self.lstm = nn.LSTMCell(32 * 6 * 6, 512)  # 长短期记忆
        self.critic_layer = nn.Linear(512, 1)
        self.actor_layer = nn.Linear(512, num_actions)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def forward(self, x, hx, cx):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        hx, cx = self.lstm(x.view(x.size(0), -1), (hx, cx))
        return self.actor_layer(hx), self.critic_layer(cx), hx, cx
