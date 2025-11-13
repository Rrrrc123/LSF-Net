import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import math
import importlib
import torchvision.models as models


class AutoRouter(nn.Module):
    def __init__(self, vector_len, prior_knowledge):
        super(AutoRouter, self).__init__()
        self.fc1 = nn.Linear(vector_len * 2 * 10, vector_len * 5)
        self.drop1 = nn.Dropout(0.2)
        self.act1 = nn.Tanh()
        self.fc2 = nn.Linear(vector_len * 5, 1)
        self.drop2 = nn.Dropout(0.2)
        self.activate = nn.Softmax(dim=-2)
        self.prior_knowledge = torch.unsqueeze(prior_knowledge.cuda(),-1)

    def forward(self, x):
        b, c, l = x.size()
        x = torch.transpose(x, 0, 1)
        x = x.reshape(c, b*l)
        x = self.prior_knowledge * x
        x = x.reshape(c, b, l)
        x = torch.transpose(x, 0, 1)
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = self.activate(x)
        return x

    def _initialize_weights(self):
        # 使用均匀分布初始化权重
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))  # Kaiming 初始化
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # 偏置初始化为0
