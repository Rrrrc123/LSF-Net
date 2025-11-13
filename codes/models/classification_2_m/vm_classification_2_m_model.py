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
    def __init__(self,prior_knowledge):
        super(Classification2MHead, self).__init__()
        self.router = AutoRouter(VECTOR_LEN,prior_knowledge)
        self.models_fusion = FeatureFusion(14, 6, 1024, 8)
        self.conv = nn.Conv2d(6, 3, 3, 1, 1)
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # self.model.conv1 = nn.Conv2d(1, 64, 3, 1, 1 , bias=False)
        self.model.fc = nn.Sequential(nn.Linear(self.model.fc.in_features, 256),
                                      nn.Dropout(0.2),
                                      nn.Linear(256, 32),
                                      nn.Dropout(0.2),
                                      nn.Linear(32, 2),
                                      # nn.Sigmoid())
                                      nn.Softmax(dim=-1))

    def forward(self, x, x_c):
        b, c, l = x.size()
        # print(b, c, l)   # 16 5 10240
        r = self.router(x)
        # r = r.reshape(b, 5, 1)
        # print(r.size())  # torch.Size([16, 5, 1])
        # sys.exit()
        y = r * x
        y = self.models_fusion(y, x_c)
        # print(r.size())
        # sys.exit()
        # for bi in range(b):
        #     for ci in range(c):
        #         y[bi, ci, :] = r[bi, ci] * y[bi, ci, :]
        # print(y.size())
        # sys.exit()
        # y = y.reshape(b, 6, 32 * 5, 32 * 2)
        y = y.reshape(b, 6, 32, 32)
        y = self.conv(y)
        # print(y.size())
        # y = y.repeat(1, 3, 1, 1)
        # y = y.cuda()
        # print(y.size())
        # sys.exit()
        y = self.model(y)
        # print(y.size())
        # sys.exit()
        return y


class Classification2MModel(nn.Module):
    def __init__(self, package_name, module_name, class_name, prior_knowledge):
        super(Classification2MModel, self).__init__()
        module = importlib.import_module(f"{package_name}.{module_name}")
        obj = getattr(module, class_name)
        self.base_model = obj()
        self.model = Classification2MHead(prior_knowledge)

    def load_base_model(self, model_path, model_head):
        self.base_model.load_models(model_path, model_head)
        self.base_model.eval()

    def forward(self, x, x_c, mask_code):
        _, en_s_list, en_i_list, _ = self.base_model(x, mask_code)
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
        return y

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
