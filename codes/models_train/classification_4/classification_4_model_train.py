import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
from codes.datas.classification_4.r_dataset_load_mat_4 import *
from codes.models.classification_4.vm_classification_4_model import *
from codes.utils.evaluation_function.classification_evaluation import *
from codes.utils.r_common import *
from codes.utils.log_save import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import wandb
import warnings
from torch.optim import lr_scheduler
import math
from copy import deepcopy
from collections import OrderedDict

os.environ["WANDB_LOG_LEVEL"] = "error"
warnings.filterwarnings("ignore")  # , module="wandb"
os.environ["WANDB_SILENT"] = "true"


class OptimizerParamClass:
    def __init__(self):
        self.is_change_lr = 1
        self.lr = 0.0005
        self.lr_policy = 'lambda_exp'
        self.niter = 12
        self.niter_decay = 5
        self.warmup_epochs = 10
        self.lr_decay = 0.99
        self.lr_decay_iters = 5
        self.fold_num_list = ['0', '1', '2', '3', '4']  #
        self.base_models_fold_num_list = ['4', '4', '4', '4', '4']  #
        self.fold_num = self.fold_num_list[0]
        self.base_models_fold_num = self.base_models_fold_num_list[0]
        self.epoch_count = 200
        self.batch_size = 16
        self.min_acc = 0.8  # 保存模型的最小acc
        self.package_name = 'codes.models.foundation_model'
        self.model_name = "vm_S_cnn_tf_T_mamba"
        self.class_name = 'MulModalityMaeMoe'
        self.modal_prob_list = [1, 1, 1, 1]
        self.prior_knowledge = torch.tensor([1, 0.5, 0.2, 0.18, 0.12])

        self.is_train_log = 0
        self.seed = 1234
        self.datasets_path = r"D:\Dataset\key_frame_10_all_mat_5_c_4_zy_a"
        self.models_save_path = "../../../checkpoints/classification_4/" + self.model_name  # 模型保存路径
        self.results_save_path = "../../../results/classification_4/" + self.model_name  # 模型保存路径
        self.wandb_project = ""
        self.classification_models_head = ""
        self.base_models_load_path = "../../../checkpoints/foundation_model/" + self.model_name
        self.base_models_head = ""
        self.base_models_tail = "_new_all"
        # self.base_models_tail = "_10_all"
        self.train_txt_file = ""
        self.test_txt_file = ""
        self.is_load_models = 0
        self.is_wandb = 0

    def update_opt(self, fold_idx):
        self.wandb_project = (
                self.model_name +
                "_classification" +
                "_20250105" +
                # ("_D_%d" % self.is_dis) +
                # ("_M_%d" % self.is_moe) +
                # ("_L_%d" % self.is_auto_route) +
                ("_F_%s" % self.fold_num_list[fold_idx])
        )  # 模型头字符串
        self.classification_models_head = (
                self.model_name +
                "_classification" +
                # ("_D_%d" % self.is_dis) +
                # ("_M_%d" % self.is_moe) +
                # ("_L_%d" % self.is_auto_route) +
                ("_F_%s" % self.fold_num_list[fold_idx])
        )  # 模型头字符串
        self.base_models_head = (
                self.model_name +
                # ("_D_%d" % self.is_dis) +
                # ("_M_%d" % self.is_moe) +
                ("_F_%s" % self.base_models_fold_num_list[fold_idx])
        )  # 模型头字符串
        self.train_txt_file = 'train_' + self.fold_num_list[fold_idx] + '.txt'
        self.test_txt_file = 'test_' + self.fold_num_list[fold_idx] + '.txt'
        if not os.path.exists(self.models_save_path):
            os.makedirs(self.models_save_path)
        if not os.path.exists(self.results_save_path):
            os.makedirs(self.results_save_path)


opt = OptimizerParamClass()
if opt.is_wandb == 1:
    wandb.login(key="2f4d2fd52729aa8dac770625c2657fe70f1073d5")
    wandb.init(project=opt.wandb_project,
               config={"epochs": opt.epoch_count, "lr": opt.lr, "batch_size": opt.batch_size})


@torch.no_grad()
def update_ema(ema_model, model, decay=0.99):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider app
        #  lying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def get_scheduler():
    global epoch, optimizer, opt

    """
    scheduler definition
    :param optimizer:  原始优化器
    :param opt: 对应参数
    :return: 对应scheduler
    """
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

        # Return the LambdaLR scheduler
    elif opt.lr_policy == 'lambda_exp':
        def lambda_rule(epoch):
            if epoch < opt.warmup_epochs:
                # 预热阶段：学习率从 0 增加到 1
                lr_l = min(1, (epoch + 1) / opt.warmup_epochs)
            else:
                # 衰减阶段：指数衰减，最小学习率为 min_lr
                lr_l = max(0.02, 1.0 * (opt.lr_decay ** (epoch - opt.warmup_epochs)))
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    elif opt.lr_policy == 'lambda_cosine':
        def lambda_rule(epoch):
            if epoch < opt.warmup_epochs:
                lr_l = epoch / opt.warmup_epochs
            else:
                lr_l = max(1e-5, 0.5 * \
                           (1. + math.cos(
                               math.pi * (epoch - opt.warmup_epochs) / (opt.epoch_count - opt.warmup_epochs))))
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epoch_count, eta_min=0)
    elif opt.lr_policy == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=opt.lr_decay)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


if __name__ == '__main__':
    set_seed(opt.seed)
    datasets = RDatasetFiveFoldOneModelityClass()
    test_datasets = RDatasetFiveFoldOneModelityClass()
    datasets.set_modality_name("ALL")
    test_datasets.set_modality_name("ALL")
    datasets_path = opt.datasets_path
    test_log_csv = RLogSave()
    train_log_csv = RLogSave()
    mpe = MulModalMulClassificationEvaluation()

    print("训练的模型名是：", opt.model_name)
    for fold_num in range(len(opt.fold_num_list)):
        opt.update_opt(fold_num)
        print("opt.classification_models_head:", opt.classification_models_head)
        models_save_path = opt.models_save_path
        models_head = opt.classification_models_head
        base_models_name = opt.base_models_head + opt.base_models_tail
        datasets.create_dataset(datasets_path, opt.train_txt_file)
        test_datasets.create_dataset(datasets_path, opt.test_txt_file)

        print(datasets.get_dataset_count())
        print(test_datasets.get_dataset_count())

        batch_size = opt.batch_size
        train_data_loader = DataLoader(datasets, batch_size=batch_size, shuffle=True, num_workers=1)
        test_data_loader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=1)

        criterion = nn.CrossEntropyLoss()
        model = ClassificationModel(opt.package_name, opt.model_name, opt.class_name, opt.prior_knowledge)
        model.load_base_model(opt.base_models_load_path, base_models_name)
        model = model.cuda()
        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

        lr = opt.lr
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.9999), eps=1e-08,
                                      weight_decay=0.0001, amsgrad=False)
        num_epochs = opt.epoch_count
        min_loss = 100
        epoch = 0
        test_log_csv.clear_all()
        train_log_csv.clear_all()
        log_table_head = ["epoch", "loss",
                          "t_acc", "l_acc", "u_acc",
                          "t_f1s", "l_f1s", "u_f1s",
                          "t_sen", "l_sen", "u_sen",
                          "t_spe", "l_spe", "u_spe",
                          "t_auc", "l_auc", "u_auc",
                          "t_pre", "l_pre", "u_pre"]
        test_log_csv.set_table_head(log_table_head)
        train_log_csv.set_table_head(log_table_head)
        scheduler = get_scheduler()
        for epoch in range(num_epochs):
            run_loss = 0
            model.freeze_base_model()
            model.set_train_mode()
            for i, data in enumerate(train_data_loader, 0):
                # if i % 5 == 0:
                #     model.thaw_auto_router()
                # else:
                #     model.freeze_auto_router()
                inputs = data['inputs']  # us_inputs,ue_inputs,cd_inputs,ce_inputs
                labels = data['labels']
                clinical = data['clinical']
                inputs = torch.transpose(inputs, 1, 2)
                # labels = torch.transpose(labels, 1, 2)
                inputs = inputs.cuda()
                labels = labels.cuda()
                clinical = clinical.cuda()
                # if epoch > 9:
                #     mask_code = mask_code_generator(4)
                # else:
                #     mask_code = [1, 1, 1, 1]
                mask_code = [1, 1, 1, 1]
                # mask_code = mask_code_generator(4, opt.modal_prob_list)
                optimizer.zero_grad()
                outputs = model(inputs, clinical, mask_code)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # update_ema(ema, model)
                run_loss += loss.item()
            train_loss = run_loss / len(train_data_loader)
            if opt.is_change_lr == 1:
                scheduler.step()
                cur_lr = scheduler.get_last_lr()[0]
            else:
                cur_lr = lr
            print("lr:", cur_lr)
            print(epoch, train_loss)
            if opt.is_wandb == 1:
                wandb.log(
                    {"epoch": epoch, "lr": cur_lr, "loss_sum": train_loss})
            if opt.is_train_log == 1:
                print("train_result:")
                log_row = mpe.get_models_performance_evaluation(model, datasets, 4, opt.modal_prob_list)
                mpe.print_specificity_list()
                mpe.print_sensitivity_list()
                mpe.print_confusion_matrix()

                log_row = [epoch] + [train_loss] + log_row
                train_log_csv.add_tail(log_row)
                train_log_csv.save_log(opt.results_save_path, "train_" + models_head)

            print("test_result:")
            log_row = mpe.get_models_performance_evaluation(model, test_datasets, 4, opt.modal_prob_list)
            mpe.print_specificity_list()
            mpe.print_sensitivity_list()
            mpe.print_confusion_matrix()

            log_row = [epoch] + [train_loss] + log_row
            test_log_csv.add_tail(log_row)
            test_log_csv.save_log(opt.results_save_path, "test_" + models_head)

            if log_row[3] > opt.min_acc:  # acc
                model.save_classification_model(opt.models_save_path,
                                                models_head + "_%03d_%.4f" % (epoch, train_loss))
