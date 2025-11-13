import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import math
from codes.models.utils.vm_auto_router import *
from codes.models.utils.vm_models_fusion import *
import importlib
import torchvision.models as models

VECTOR_LEN = 512


class Classification2MHead(nn.Module):
    def __init__(self, is_auto_route, is_models_fusion):
        super(Classification2MHead, self).__init__()
        self.router = AutoRouter(VECTOR_LEN)
        self.models_fusion = FeatureFusion(14, 6, 1024, 8)
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # self.model.conv1 = nn.Conv2d(1, 64, 3, 1, 1 , bias=False)
        self.model.fc = nn.Sequential(nn.Linear(self.model.fc.in_features, 256),
                                      nn.Dropout(0.2),
                                      nn.Linear(256, 32),
                                      nn.Dropout(0.2),
                                      nn.Linear(32, 2),
                                      # nn.Sigmoid())
                                      nn.Softmax(dim=-1))
        self.is_auto_route = is_auto_route
        self.is_models_fusion = is_models_fusion
        if is_models_fusion == 1:
            self.conv = nn.Conv2d(6, 3, 3, 1, 1)
        else:
            self.liner2 = nn.Linear(10240, 1024)
            self.conv = nn.Conv2d(5, 3, 3, 1, 1)

    def forward(self, x, x_c):
        b, c, l = x.size()
        if self.is_auto_route == 1:
            r = self.router(x)
            y = r * x
        else:
            y = x
        if self.is_models_fusion:
            y = self.models_fusion(y, x_c)
            # print(y.size())
            y = y.reshape(b, 6, 32, 32)
        else:
            # y = self.models_fusion(y, x_c)
            y = self.liner2(y)
            # print(y.size())  #  16  6  1024
            y = y.reshape(b, 5, 32, 32)
        y = self.conv(y)
        y = self.model(y)
        return y


class Classification2MModel(nn.Module):
    def __init__(self, package_name, module_name, class_name, is_moe, is_auto_route, is_models_fusion):
        super(Classification2MModel, self).__init__()
        module = importlib.import_module(f"{package_name}.{module_name}")
        obj = getattr(module, class_name)
        self.base_model = obj(is_moe)
        self.model = Classification2MHead(is_auto_route, is_models_fusion)

    def load_base_model(self, model_path, model_head):
        self.base_model.load_models(model_path, model_head)
        self.base_model.eval()

    def forward(self, x, x_c, mask_code):
        _, en_s_list, en_i_list = self.base_model(x, mask_code)
        b, c, l = en_i_list.size()
        en_s_list = en_s_list.reshape(b, c // 10, l * 10)
        en_i_list = en_i_list.reshape(b, c // 10, l * 10)
        b, c, l = en_i_list.size()
        # print(en_s_list.size())
        # print(en_i_list.size())
        en_s_list = torch.mean(en_s_list, dim=-2)
        en_s_list = en_s_list.reshape(b, 1, l)
        y = torch.cat((en_s_list, en_i_list), dim=1)
        # print(y.size())
        # sys.exit()
        y = y.cuda()
        y = self.model(y, x_c)
        # print(y.size())
        # sys.exit()
        return y, en_s_list, en_i_list

    def set_train_mode(self):
        self.base_model.eval()
        self.model.train()

    def set_eval_mode(self):
        self.base_model.eval()
        self.model.eval()

    def freeze_base_model(self):
        for param in self.base_model.parameters():
            param.requires_grad = False  # 冻结所有参数

    def freeze_auto_router(self):
        for param in self.router.parameters():
            param.requires_grad = False  # 冻结所有参数

    def thaw_auto_router(self):
        for param in self.router.parameters():
            param.requires_grad = True  # 冻结所有参数

    def save_model(self, file_path, file_head):
        save_dir = os.path.join(file_path, file_head + ".pth")
        print("save_model:", save_dir)
        torch.save(self.state_dict(), save_dir)

    def save_classification_model(self, file_path, file_head):
        save_dir = os.path.join(file_path, file_head + "_h.pth")
        print("save_model:", save_dir)
        torch.save(self.model.state_dict(), save_dir)

    def load_classification_model(self, file_path, file_head):
        load_dir = os.path.join(file_path, file_head + "_h.pth")
        print("load_model:", load_dir)
        self.model.load_state_dict(torch.load(load_dir))

    def load_model(self, file_path, file_head):
        load_dir = os.path.join(file_path, file_head + ".pth")
        print("load_model:", load_dir)
        self.load_state_dict(torch.load(load_dir))
