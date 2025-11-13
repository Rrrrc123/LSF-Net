# coding:utf-8
# File:       r_dataset_load.py
# Author:     tfsy-rifnyga
# Created on: 2024-11-25
# function  实现各个病例的UE图像数据加载,并生成mat文件。输入图像数值范围-1~1


import os
import sys
import copy

import torch
from torch.utils.data import Dataset
import json

import numpy as np
import scipy.io as io


class RDatasetFiveFoldOneModelityClass(Dataset):
    def __init__(self):
        self.us_samples_list = []
        self.us_dataset_path = ""
        self.modality_name = 'US'

    def __len__(self):
        return len(self.us_samples_list)

    def set_modality_name(self, name):
        self.modality_name = name

    def __getitem__(self, idx):
        mat_file = self.us_samples_list[idx]
        # print(">>>>:", mat_file)
        data = io.loadmat(mat_file)
        # print(data)
        # sys.exit()
        inputs = torch.tensor(data[self.modality_name]/255.0 * 2 - 1, dtype=torch.float32)
        # inputs = torch.tensor(data[self.modality_name] / 255.0, dtype=torch.float32)
        labels = torch.tensor(data['label'][0]*1.0, dtype=torch.float32)
        seg_labels = torch.tensor(data['seg_label'] / 255.0, dtype=torch.float32)
        return {'inputs': inputs, 'labels': labels, 'seg_labels': seg_labels}

    def get_dataset_count(self):
        return len(self.us_samples_list)

    def create_dataset(self, train_dir, train_txt):
        self.us_dataset_path = os.path.join(train_dir, "data")
        txt_path = os.path.join(train_dir, train_txt)
        with open(txt_path, 'r') as file:
            lines = file.readlines()
        us_samples_list = [line.strip() for line in lines]
        self.us_samples_list = []
        for mat_file in us_samples_list:
            self.us_samples_list.append(os.path.join(train_dir, "data", mat_file))
        print(self.us_samples_list)

