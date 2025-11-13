import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import math
import torch.nn.init as init
from einops.layers.torch import Rearrange
import importlib
import torchvision.models as models
from mamba_ssm import Mamba

VECTOR_LEN = 512


class ClinicalInfToVector(nn.Module):
    def __init__(self, clinical_nf_len, vector_len):
        super(ClinicalInfToVector, self).__init__()
        self.liner1 = nn.Linear(clinical_nf_len, vector_len//16)
        self.liner2 = nn.Linear(vector_len//16, vector_len)
        self.dropout = nn.Dropout(0.2)
        self.activite = nn.Tanh()

    def forward(self, x):
        x = self.liner1(x)
        x = self.dropout(x)
        x = self.liner2(x)
        x = self.dropout(x)
        x = self.activite(x)
        return x


class VmFusionBlocker(nn.Module):
    def __init__(self, input_size):
        super(VmFusionBlocker, self).__init__()
        self.mamba = Mamba(  # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=1024,  # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
        )
        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(10240, 1024)
        self.activate = nn.Tanh()

    def forward(self, x):
        b,c,l = x.size()  # [16, 5, 10240]
        # print(x.size())
        # sys.exit()
        x = x.reshape((b, c*10, l//10))  # [16, 50, 1024]
        x = self.mamba(x)
        x = x.reshape((b, c, l))  # [16, 5, 10240]
        # print(x.size())
        # sys.exit()
        # lstm_out: torch.Size([4, 10, 1024]), h_n: torch.Size([10, 4, 1024]), c_n: torch.Size([10, 4, 1024])
        x = self.fc1(x)  # [16, 5, 1024]
        x = self.dropout1(x)
        x = self.activate(x)# [16, 5, 1024]
        # print(x.size())
        # sys.exit()
        return x


class VmClinicalFusionBlocker(nn.Module):
    def __init__(self, in_c, embed_dim, n_heads=8):
        super(VmClinicalFusionBlocker, self).__init__()
        self.mul_head_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads)
        self.nb = nn.BatchNorm1d(in_c)
        self.activate = nn.Tanh()

    def forward(self, x):
        residual_x = x
        x = torch.transpose(x, 0, 1)
        x = self.mul_head_attn(x, x, x)[0]
        x = torch.transpose(x, 0, 1)
        x = x + residual_x
        x = self.nb(x)
        x = self.activate(x)
        # print(x.size())  # torch.Size([16, 6, 1024])
        # sys.exit()
        return x

    def _initialize_weights(self):
        # 使用均匀分布初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)


class FeatureFusion(nn.Module):
    def __init__(self, input_size, in_c, embed_dim, n_heads=8):
        super(FeatureFusion, self).__init__()
        # self.bn1d = nn.BatchNorm1d(1)
        self.clinical_liner = ClinicalInfToVector(input_size, 1024)
        self.vm_fusion = VmFusionBlocker(1024)
        self.vm_clinical_fusion = VmClinicalFusionBlocker(in_c, embed_dim, n_heads)

    def forward(self, x, x_c):
        # print(x_c.size())
        # x_c = self.bn1d(x_c)
        x_c = self.clinical_liner(x_c)
        y = self.vm_fusion(x)
        # print(x_c.size(), y.size())
        # sys.exit()
        y = torch.cat((y, x_c), dim=1)
        # print(y.size())
        # sys.exit()
        y = self.vm_clinical_fusion(y)
        return y


