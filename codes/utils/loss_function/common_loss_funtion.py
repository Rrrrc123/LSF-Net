import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.models.video as models
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report
from torchvision.models.video import R3D_18_Weights
import wandb
import warnings
from torch.optim import lr_scheduler
import math
from copy import deepcopy
from collections import OrderedDict
import random



def contrastive_loss(output1, output2, labels_1, margin=1.0):
    # label = 1
    # 表示这对样本是正样本对（即它们属于同一类）。
    # label = 0
    # 表示这对样本是负样本对（即它们属于不同类）。
    distance = nn.functional.pairwise_distance(output1, output2)
    loss_1 = (1 - labels_1) * torch.pow(distance, 2) + labels_1 * torch.pow(torch.clamp(margin - distance, min=0.0), 2)
    return loss_1.mean()


def cosine_similarity_matrix(vector2):
    """
    计算多个向量的余弦相似度矩阵
    :param vectors: 向量列表（二维数组）
    :return: 余弦相似度矩阵
    """
    # 将输入列表转换为 PyTorch 张量
    vectors = vector2.clone().detach()
    # 计算余弦相似度矩阵
    similarity_matrix = torch.nn.functional.cosine_similarity(vectors.unsqueeze(1), vectors.unsqueeze(0), dim=2)

    return similarity_matrix ** 2


def cosine_similarity(vec_a, vec_b):
    """
    计算两个向量的余弦相似度
    :param vec_a: 第一个向量 (可以是列表或 PyTorch 张量)
    :param vec_b: 第二个向量 (可以是列表或 PyTorch 张量)
    :return: 余弦相似度
    """
    # 将输入向量转换为 PyTorch 张量
    # a = vec_a.clone().detach()
    # b = vec_b.clone().detach()
    a = vec_a
    b = vec_b
    # print(a.size(), b.size())

    # 计算余弦相似度
    cos_sim = torch.nn.functional.cosine_similarity(a, b, dim=0)
    return cos_sim ** 2  # 返回标量值


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=1):
        """
        Implementation of the loss described in the paper Supervised Contrastive Learning :
        https://arxiv.org/abs/2004.11362

        :param temperature: int
        """
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, projections, targets):  # targets mask
        """

        :param projections: torch.Tensor, shape [batch_size, projection_dim]
        :param targets: torch.Tensor, shape [batch_size]
        :return: torch.Tensor, scalar
        """
        device = torch.device("cuda") if projections.is_cuda else torch.device("cpu")

        dot_product_tempered = torch.mm(projections, projections.T) / self.temperature
        for m_i in range(8):
            dot_product_tempered[m_i][m_i] = 0
        # print(dot_product_tempered.size())
        # sys.exit()
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = (
                torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )
        # exp_dot_tempered = (
        #         torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        # )
        # print('exp_dot_tempered:', exp_dot_tempered)
        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device)
        # print('mask_similar_class:', mask_similar_class)

        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device)
        # print('mask_anchor_out:\n', mask_anchor_out)
        mask_combined = mask_similar_class * mask_anchor_out + 1e-5
        # print('mask_combined:\n', mask_combined)
        cardinality_per_samples = torch.sum(mask_combined, dim=1)
        # print('cardinality_per_samples:\n', cardinality_per_samples)
        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
        # print('log_prob:\n', log_prob)
        supervised_contrastive_loss = torch.mean(log_prob * mask_combined)
        return supervised_contrastive_loss


# class SupervisedContrastiveLoss(nn.Module):
#     def __init__(self, temperature=1):
#         """
#         Implementation of the loss described in the paper Supervised Contrastive Learning :
#         https://arxiv.org/abs/2004.11362
#
#         :param temperature: int
#         """
#         super(SupervisedContrastiveLoss, self).__init__()
#         self.temperature = temperature
#
#     def forward(self, projections, targets):  # targets mask
#         """
#
#         :param projections: torch.Tensor, shape [batch_size, projection_dim]
#         :param targets: torch.Tensor, shape [batch_size]
#         :return: torch.Tensor, scalar
#         """
#         device = torch.device("cuda") if projections.is_cuda else torch.device("cpu")
#
#         dot_product_tempered = torch.mm(projections, projections.T) / self.temperature
#         # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
#         exp_dot_tempered = (
#                 torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
#         )
#         # print('exp_dot_tempered:', exp_dot_tempered)
#         mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(device)
#         # print('mask_similar_class:', mask_similar_class)
#
#         mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(device)
#         # print('mask_anchor_out:\n', mask_anchor_out)
#         mask_combined = mask_similar_class * mask_anchor_out + 1e-5
#         # print('mask_combined:\n', mask_combined)
#         cardinality_per_samples = torch.sum(mask_combined, dim=1)
#         # print('cardinality_per_samples:\n', cardinality_per_samples)
#         log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
#         # print('log_prob:\n', log_prob)
#         supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples
#         # print('supervised_contrastive_loss_per_sample:', supervised_contrastive_loss_per_sample)
#         supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)
#         # print('supervised_contrastive_loss:', supervised_contrastive_loss)
#         # sys.exit()
#         return supervised_contrastive_loss