import numpy as np
import torch
from torch.utils.data import DataLoader
from codes.utils.r_common import *
from sklearn.metrics import (
    matthews_corrcoef,  # 计算MCC
    balanced_accuracy_score,  # 计算平衡准确率
    f1_score,  # 计算F1分数
    recall_score,  # 计算召回率
    roc_auc_score,  # 计算AUC
    average_precision_score,  # 计算平均精确度
    precision_score,  # 计算精确率
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
import scikits.bootstrap as bootstrap
from sklearn.utils import resample
import seaborn as sns


def bootstrap_confidence_interval(data, confidence_level=0.95):
    # bootstrap_means = np.mean(data)
    lower_bound = np.percentile(np.sort(data), (1 - confidence_level) / 2 * 100)  # (1 - confidence_level) / 2 * 100
    upper_bound = np.percentile(np.sort(data), (1 + confidence_level) / 2 * 100)  # (1 + confidence_level) / 2 * 100
    # print(bootstrap_means, lower_bound, upper_bound)
    return lower_bound, upper_bound


def auc_confidence_interval(y_true, y_pred_proba, confidence_level=0.95, n_iterations=1000):
    aucs = []
    for _ in range(n_iterations):
        indices = resample(range(len(y_true)), replace=True)
        fpr, tpr, thresholds = roc_curve(np.array(y_true)[indices], np.array(y_pred_proba)[indices])
        aucs.append(auc(fpr, tpr))

    lower_bound = np.percentile(aucs, ((1 - confidence_level) / 2) * 100)
    upper_bound = np.percentile(aucs, (confidence_level + (1 - confidence_level) / 2) * 100)
    return lower_bound, upper_bound


class MulModalMulClassificationEvaluation:
    def __init__(self):
        self.confusion_matrix = []
        self.performance_evaluation = []
        self.classification_report = []
        self.specificity_list = []
        self.sensitivity_list = []
        self.classification_num = 5
        self.modal_prob_list = [1.0, 0.5, 0.7, 0.3]

    def get_models_performance_evaluation(self, t_model, t_datasets, classification_num, modal_prob_list):
        self.modal_prob_list = modal_prob_list
        self.classification_num = classification_num
        datasets_len = t_datasets.get_dataset_count()
        data_loader = DataLoader(t_datasets, batch_size=1, shuffle=False, num_workers=0)
        y_pred_prob = np.zeros((datasets_len, classification_num))  # 5分类的概率值
        y_pred = np.zeros(datasets_len)  # 预测类别 # [2,3,4,1,0]
        y_labels = np.zeros(datasets_len)  # 原始标签的类别 # [1,2,3,4,0]
        t_model.eval()
        train_correct_sum = 0
        # train_simple_cnt = datasets_len
        with torch.no_grad():
            for i, data in enumerate(data_loader, 0):
                inputs = data['inputs']
                labels = data['labels']
                clinical = data['clinical']
                inputs = torch.transpose(inputs, 1, 2)
                inputs = inputs.cuda()
                labels = labels.cuda()
                clinical = clinical.cuda()
                mask_code = [1, 1, 1, 1]
                # mask_code = mask_code_generator(len(modal_prob_list), modal_prob_list)
                outputs = t_model(inputs, clinical, mask_code)
                y_pred_prob[i] = outputs[0].cpu().numpy()
                y_pred[i] = np.argmax(y_pred_prob[i], axis=0)
                y_labels[i] = np.argmax(labels[0].cpu().numpy(), axis=0)
                if y_pred[i] == y_labels[i]:
                    train_correct_sum += 1

        num_iterations = 1000
        # index_list = np.linspace(0, len(y_pred) - 1, len(y_pred), dtype=np.int32)
        # print(index_list)
        acc_list = np.zeros(num_iterations)
        auc_list = np.zeros(num_iterations)
        f1s_list = np.zeros(num_iterations)
        spe_list = np.zeros(num_iterations)
        sen_list = np.zeros(num_iterations)
        pre_list = np.zeros(num_iterations)
        for i in range(num_iterations):
            choice_index_list = np.random.choice(len(y_pred), size=len(y_pred), replace=True)
            acc_list[i] = accuracy_score(y_labels[choice_index_list], y_pred[choice_index_list])
            f1s_list[i] = f1_score(y_labels[choice_index_list], y_pred[choice_index_list], average='weighted')
            specificity_list = []
            sensitivity_list = []
            conf_matrix = confusion_matrix(y_labels[choice_index_list], y_pred[choice_index_list])
            for i in range(len(conf_matrix)):
                TP = conf_matrix[i, i]  # True Positive
                FN = np.sum(conf_matrix[i, :]) - TP  # False Negative
                FP = np.sum(conf_matrix[:, i]) - TP  # False Positive
                TN = np.sum(conf_matrix) - (TP + FN + FP)  # True Negative

                SEN = TP / (TP + FN) if (TP + FN) > 0 else 0  # 灵敏度 (Sensitivity)
                SPE = TN / (TN + FP) if (TN + FP) > 0 else 0  # 特异性 (Specificity)

                sensitivity_list.append(SEN)
                specificity_list.append(SPE)
            sen_list[i] = np.mean(sensitivity_list)
            spe_list[i] = np.mean(specificity_list)
            # choice_index_list = np.random.choice(index_list, size=len(index_list), replace=True)
            auc_list[i] = roc_auc_score(y_labels[choice_index_list], y_pred_prob[choice_index_list],
                                        average='weighted', multi_class='ovr')
            # choice_index_list = np.random.choice(index_list, size=len(index_list), replace=True)
            pre_list[i] = precision_score(y_labels[choice_index_list], y_pred[choice_index_list],
                                          average='weighted')

        acc_lower, acc_upper = bootstrap_confidence_interval(acc_list)
        f1s_lower, f1s_upper = bootstrap_confidence_interval(f1s_list)
        spe_lower, spe_upper = bootstrap_confidence_interval(spe_list)
        auc_lower, auc_upper = bootstrap_confidence_interval(auc_list)
        sen_lower, sen_upper = bootstrap_confidence_interval(sen_list)
        pre_lower, pre_upper = bootstrap_confidence_interval(pre_list)

        t_auc = roc_auc_score(y_labels, y_pred_prob, average='weighted', multi_class='ovr')
        t_acc = accuracy_score(y_labels, y_pred)
        t_f1s = f1_score(y_labels, y_pred, average='weighted')  # 加权平均
        t_pre = precision_score(y_labels, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_labels, y_pred)
        self.confusion_matrix = conf_matrix
        specificity_list = []
        sensitivity_list = []
        for i in range(len(conf_matrix)):
            TP = conf_matrix[i, i]  # True Positive
            FN = np.sum(conf_matrix[i, :]) - TP  # False Negative
            FP = np.sum(conf_matrix[:, i]) - TP  # False Positive
            TN = np.sum(conf_matrix) - (TP + FN + FP)  # True Negative

            SEN = TP / (TP + FN) if (TP + FN) > 0 else 0  # 灵敏度 (Sensitivity)
            SPE = TN / (TN + FP) if (TN + FP) > 0 else 0  # 特异性 (Specificity)

            sensitivity_list.append(SEN)
            specificity_list.append(SPE)
        self.sensitivity_list = sensitivity_list
        self.specificity_list = specificity_list

        self.classification_report = classification_report(y_labels, y_pred)

        self.performance_evaluation = [
            t_acc, acc_lower, acc_upper,
            t_f1s, f1s_lower, f1s_upper,
            np.mean(sensitivity_list), sen_lower, sen_upper,
            np.mean(specificity_list), spe_lower, spe_upper,
            t_auc, auc_lower, auc_upper,
            t_pre, pre_lower, pre_upper]
        # print(pe)
        self.print_performance_evaluation()
        return self.performance_evaluation

    def print_performance_evaluation(self):
        print(
            "      t_acc                  t_f1_scor               t_sen                   t_spe                   t_auc                  t_prec  ")
        print("%.4f(%.4f,%.4f)  %.4f(%.4f,%.4f)  %.4f(%.4f,%.4f)  %.4f(%.4f,%.4f)  %.4f(%.4f,%.4f)  %.4f(%.4f,%.4f)" % (
            self.performance_evaluation[0], self.performance_evaluation[1], self.performance_evaluation[2],
            self.performance_evaluation[3], self.performance_evaluation[4], self.performance_evaluation[5],
            self.performance_evaluation[6], self.performance_evaluation[7], self.performance_evaluation[8],
            self.performance_evaluation[9], self.performance_evaluation[10], self.performance_evaluation[11],
            self.performance_evaluation[12], self.performance_evaluation[13], self.performance_evaluation[14],
            self.performance_evaluation[15], self.performance_evaluation[16], self.performance_evaluation[17]))

    def print_confusion_matrix(self):
        print("混淆矩阵:")
        print(self.confusion_matrix)

    def get_confusion_matrix_hot_map(self, classification_labels):
        # classification_labels = ['Class 0', 'Class 1', 'Class 2', 'Class 3']
        print("混淆矩阵:")
        print(self.confusion_matrix)
        plt.figure(figsize=(8, 6))
        self.confusion_matrix = np.transpose(self.confusion_matrix)
        self.confusion_matrix = self.confusion_matrix / np.sum(self.confusion_matrix, axis=0)
        self.confusion_matrix = np.transpose(self.confusion_matrix)
        sns.heatmap(self.confusion_matrix, annot=True, fmt='.1%', cmap='Blues', cbar=True,
                    xticklabels=classification_labels,
                    yticklabels=classification_labels)
        plt.ylabel('Ground Truth')
        plt.xlabel('Prediction')
        plt.title('Confusion Matrix Heatmap')
        plt.show()

    def print_sensitivity_list(self):
        print("每个类别的灵敏度 (SEN):")
        print(self.sensitivity_list)

    def print_specificity_list(self):
        print("每个类别的特异性 (SPE):")
        print(self.specificity_list)

    def print_classification_report(self):
        print("分类报告:")
        print(self.classification_report)


class MulModalBinClassificationEvaluation:
    def __init__(self):
        self.confusion_matrix = []
        self.performance_evaluation = []
        self.classification_report = []
        self.specificity_list = []
        self.sensitivity_list = []
        self.classification_num = 2
        self.modal_prob_list = [1.0, 0.5, 0.7, 0.3]


    def get_models_performance_evaluation(self, t_model, t_datasets, positive_sample_idx=1):
        datasets_len = t_datasets.get_dataset_count()
        data_loader = DataLoader(t_datasets, batch_size=1, shuffle=False, num_workers=0)
        y_raw_prob = np.zeros((datasets_len, 2))  # 2分类的原始值
        y_pred = np.zeros(datasets_len)  # 预测类别 # [2,3,4,1,0]
        y_pred_prob = np.zeros(datasets_len)  # 2分类的概率值
        y_labels = np.zeros(datasets_len)  # 原始标签的类别 # [1,2,3,4,0]

        t_model.eval()
        with torch.no_grad():
            for i, data in enumerate(data_loader, 0):
                inputs = data['inputs']
                labels = data['labels']
                clinical = data['clinical']
                inputs = torch.transpose(inputs, 1, 2)
                inputs = inputs.cuda()
                labels = labels.cuda()
                clinical = clinical.cuda()
                mask_code = [1, 1, 1, 1]
                outputs = t_model(inputs, clinical, mask_code)
                y_raw_prob[i] = outputs[0].cpu().numpy()

                # y_pred[i] = y_raw_prob[i][positive_sample_idx] >= 0.5
                y_pred[i] = np.argmax(y_pred_prob[i])

                y_pred_prob[i] = y_raw_prob[i][positive_sample_idx]  # y_pred[i].astype(int)

                y_labels[i] = labels[0].cpu().numpy()[positive_sample_idx]


            num_iterations = 1000
            index_list = np.linspace(0, len(y_pred) - 1, len(y_pred), dtype=np.int32)
            # print(index_list)
            acc_list = np.zeros(num_iterations)
            auc_list = np.zeros(num_iterations)
            f1s_list = np.zeros(num_iterations)
            spe_list = np.zeros(num_iterations)
            sen_list = np.zeros(num_iterations)
            pre_list = np.zeros(num_iterations)
            rng = np.random.RandomState(42)
            for i in range(num_iterations):
                # choice_index_list = rng.choice(len(y_labels), size=len(y_labels), replace=True)
                choice_index_list = np.random.choice(index_list, size=len(index_list), replace=True)
                acc_list[i] = accuracy_score(y_labels[choice_index_list], y_pred[choice_index_list])
                # choice_index_list = np.random.choice(index_list, size=len(index_list), replace=True)
                f1s_list[i] = f1_score(y_labels[choice_index_list], y_pred[choice_index_list])
                # choice_index_list = np.random.choice(index_list, size=len(index_list), replace=True)
                tn, fp, fn, tp = confusion_matrix(y_labels[choice_index_list], y_pred[choice_index_list]).ravel()
                spe_list[i] = tn / (tn + fp)
                sen_list[i] = tp / (tp + fn)
                # choice_index_list = np.random.choice(index_list, size=len(index_list), replace=True)
                auc_list[i] = roc_auc_score(y_labels[choice_index_list], y_pred_prob[choice_index_list])
                # choice_index_list = np.random.choice(index_list, size=len(index_list), replace=True)
                pre_list[i] = precision_score(y_labels[choice_index_list], y_pred[choice_index_list])
            acc_lower, acc_upper = bootstrap_confidence_interval(acc_list)
            f1s_lower, f1s_upper = bootstrap_confidence_interval(f1s_list)
            spe_lower, spe_upper = bootstrap_confidence_interval(spe_list)
            auc_lower, auc_upper = bootstrap_confidence_interval(auc_list)
            sen_lower, sen_upper = bootstrap_confidence_interval(sen_list)
            pre_lower, pre_upper = bootstrap_confidence_interval(pre_list)
            tn, fp, fn, tp = confusion_matrix(y_labels, y_pred).ravel()
            self.confusion_matrix = confusion_matrix(y_labels, y_pred)
            t_acc = accuracy_score(y_labels, y_pred)
            t_f1s = f1_score(y_labels, y_pred)
            # t_recall = recall_score(y_labels, y_pred)
            t_spe = tn / (tn + fp)  # pos_label=0,
            # fpr, tpr, thresholds = roc_curve(y_labels, y_pred_prob)
            t_auc = roc_auc_score(y_labels, y_pred_prob)
            t_sen = tp / (tp + fn)
            t_pre = precision_score(y_labels, y_pred)
        self.performance_evaluation = [t_auc, auc_lower, auc_upper,
                                       t_acc, acc_lower, acc_upper,
                                       t_f1s, f1s_lower, f1s_upper,
                                       t_sen, sen_lower, sen_upper,
                                       t_spe, spe_lower, spe_upper,
                                       t_pre, pre_lower, pre_upper]
        self.print_performance_evaluation()
        # sys.exit()
        return self.performance_evaluation


    def get_models_performance_evaluation_new(self, t_model, t_datasets, positive_sample_idx=1):
        datasets_len = t_datasets.get_dataset_count()
        data_loader = DataLoader(t_datasets, batch_size=1, shuffle=False, num_workers=0)
        y_raw_prob = np.zeros((datasets_len, 2))  # 2分类的原始值
        y_pred = np.zeros(datasets_len)  # 预测类别 # [2,3,4,1,0]
        y_pred_prob = np.zeros(datasets_len)  # 2分类的概率值
        y_labels = np.zeros(datasets_len)  # 原始标签的类别 # [1,2,3,4,0]
        t_model.eval()
        with torch.no_grad():
            for i, data in enumerate(data_loader, 0):
                inputs = data['inputs']
                labels = data['labels']
                clinical = data['clinical']
                inputs = torch.transpose(inputs, 1, 2)
                inputs = inputs.cuda()
                labels = labels.cuda()
                clinical = clinical.cuda()
                mask_code = [1, 1, 1, 1]
                outputs = t_model(inputs, clinical, mask_code)
                y_raw_prob[i] = outputs[0].cpu().numpy()
                # y_pred[i] = y_raw_prob[i][positive_sample_idx] >= 0.5
                y_pred[i] = np.argmax(y_raw_prob[i], axis=0)
                y_pred_prob[i] = y_raw_prob[i][positive_sample_idx]  # y_pred[i].astype(int)
                y_labels[i] = labels[0].cpu().numpy()[positive_sample_idx]
            num_iterations = 1000
            index_list = np.linspace(0, len(y_pred) - 1, len(y_pred), dtype=np.int32)
            acc_list = np.zeros(num_iterations)
            auc_list = np.zeros(num_iterations)
            f1s_list = np.zeros(num_iterations)
            spe_list = np.zeros(num_iterations)
            sen_list = np.zeros(num_iterations)
            pre_list = np.zeros(num_iterations)
            rng = np.random.RandomState(42)
            for i in range(num_iterations):
                # choice_index_list = rng.choice(len(y_labels), size=len(y_labels), replace=True)
                choice_index_list = np.random.choice(index_list, size=len(index_list), replace=True)
                acc_list[i] = accuracy_score(y_labels[choice_index_list], y_pred[choice_index_list])
                # choice_index_list = np.random.choice(index_list, size=len(index_list), replace=True)
                f1s_list[i] = f1_score(y_labels[choice_index_list], y_pred[choice_index_list])
                # choice_index_list = np.random.choice(index_list, size=len(index_list), replace=True)
                tn, fp, fn, tp = confusion_matrix(y_labels[choice_index_list], y_pred[choice_index_list]).ravel()
                spe_list[i] = tn / (tn + fp)
                sen_list[i] = tp / (tp + fn)
                # choice_index_list = np.random.choice(index_list, size=len(index_list), replace=True)
                auc_list[i] = roc_auc_score(y_labels[choice_index_list], y_pred_prob[choice_index_list])
                # choice_index_list = np.random.choice(index_list, size=len(index_list), replace=True)
                pre_list[i] = precision_score(y_labels[choice_index_list], y_pred[choice_index_list])
            acc_lower, acc_upper = bootstrap_confidence_interval(acc_list)
            f1s_lower, f1s_upper = bootstrap_confidence_interval(f1s_list)
            spe_lower, spe_upper = bootstrap_confidence_interval(spe_list)
            auc_lower, auc_upper = bootstrap_confidence_interval(auc_list)
            sen_lower, sen_upper = bootstrap_confidence_interval(sen_list)
            pre_lower, pre_upper = bootstrap_confidence_interval(pre_list)
            tn, fp, fn, tp = confusion_matrix(y_labels, y_pred).ravel()
            self.confusion_matrix = confusion_matrix(y_labels, y_pred)
            t_acc = accuracy_score(y_labels, y_pred)
            t_f1s = f1_score(y_labels, y_pred)
            # t_recall = recall_score(y_labels, y_pred)
            t_spe = tn / (tn + fp)  # pos_label=0,
            # fpr, tpr, thresholds = roc_curve(y_labels, y_pred_prob)
            t_auc = roc_auc_score(y_labels, y_pred_prob)
            t_sen = tp / (tp + fn)
            t_pre = precision_score(y_labels, y_pred)
        self.performance_evaluation = {"AUC": t_auc, "AUC_L": auc_lower, "AUC_U": auc_upper,
                                       "ACC": t_acc, "ACC_L": acc_lower, "ACC_U": acc_upper,
                                       "F1S": t_f1s, "F1S_L": f1s_lower, "F1S_U": f1s_upper,
                                       "SEN": t_sen, "SEN_L": sen_lower, "SEN_U": sen_upper,
                                       "SPE": t_spe, "SPE_L": spe_lower, "SPE_U": spe_upper,
                                       "PRE": t_pre, "PRE_L": pre_lower, "PRE_U": pre_upper}
        self.print_performance_evaluation_new()
        # sys.exit()
        return self.performance_evaluation

    def get_models_roc_curve(self, t_model, t_datasets, doctors, markers, colors, positive_sample_idx=1):
        datasets_len = t_datasets.get_dataset_count()
        data_loader = DataLoader(t_datasets, batch_size=1, shuffle=False, num_workers=0)
        y_raw_prob = np.zeros((datasets_len, 2))  # 2分类的原始值
        y_pred = np.zeros(datasets_len)  # 预测类别 # [2,3,4,1,0]
        y_pred_prob = np.zeros(datasets_len)  # 2分类的概率值
        y_labels = np.zeros(datasets_len)  # 原始标签的类别 # [1,2,3,4,0]

        t_model.eval()
        train_correct_sum = 0
        train_simple_cnt = datasets_len
        with torch.no_grad():
            for i, data in enumerate(data_loader, 0):
                inputs = data['inputs']
                labels = data['labels']
                clinical = data['clinical']
                inputs = torch.transpose(inputs, 1, 2)
                inputs = inputs.cuda()
                labels = labels.cuda()
                clinical = clinical.cuda()
                mask_code = [1, 1, 1, 1]
                outputs = t_model(inputs, clinical, mask_code)
                y_raw_prob[i] = outputs[0].cpu().numpy()
                # print("y_raw_prob[i]:", i, y_raw_prob[i])
                y_pred[i] = np.argmax(y_raw_prob[i], axis=0)
                # print("y_pred[i]:", i, y_pred[i].astype(int))
                y_pred_prob[i] = y_raw_prob[i][positive_sample_idx]  # y_pred[i].astype(int)
                # print(i, y_pred_prob[i])
                y_labels[i] = np.argmax(labels[0].cpu().numpy(), axis=0)
                # print(i, y_labels[i])
                # if y_pred[i] == y_labels[i]:
                #     train_correct_sum += 1
        # fpr, tpr, thresholds = roc_curve(y_labels, y_pred_prob)
        # roc_auc = auc(fpr, tpr)

        num_iterations = 1000
        index_list = np.linspace(0, len(y_pred) - 1, len(y_pred), dtype=np.int32)
        # print(index_list)
        acc_list = np.zeros(num_iterations)
        auc_list = np.zeros(num_iterations)
        f1s_list = np.zeros(num_iterations)
        spe_list = np.zeros(num_iterations)
        sen_list = np.zeros(num_iterations)
        pre_list = np.zeros(num_iterations)
        rng = np.random.RandomState(42)
        for i in range(num_iterations):
            # choice_index_list = rng.choice(len(y_labels), size=len(y_labels), replace=True)
            choice_index_list = np.random.choice(index_list, size=len(index_list), replace=True)
            acc_list[i] = accuracy_score(y_labels[choice_index_list], y_pred[choice_index_list])
            # choice_index_list = np.random.choice(index_list, size=len(index_list), replace=True)
            f1s_list[i] = f1_score(y_labels[choice_index_list], y_pred[choice_index_list])
            # choice_index_list = np.random.choice(index_list, size=len(index_list), replace=True)
            tn, fp, fn, tp = confusion_matrix(y_labels[choice_index_list], y_pred[choice_index_list]).ravel()
            spe_list[i] = tn / (tn + fp)
            sen_list[i] = tp / (tp + fn)
            # choice_index_list = np.random.choice(index_list, size=len(index_list), replace=True)
            auc_list[i] = roc_auc_score(y_labels[choice_index_list], y_pred_prob[choice_index_list])
            # choice_index_list = np.random.choice(index_list, size=len(index_list), replace=True)
            pre_list[i] = precision_score(y_labels[choice_index_list], y_pred[choice_index_list])

        acc_lower, acc_upper = bootstrap_confidence_interval(acc_list)
        f1s_lower, f1s_upper = bootstrap_confidence_interval(f1s_list)
        spe_lower, spe_upper = bootstrap_confidence_interval(spe_list)
        auc_lower, auc_upper = bootstrap_confidence_interval(auc_list)
        sen_lower, sen_upper = bootstrap_confidence_interval(sen_list)
        pre_lower, pre_upper = bootstrap_confidence_interval(pre_list)
        tn, fp, fn, tp = confusion_matrix(y_labels, y_pred).ravel()
        self.confusion_matrix = confusion_matrix(y_labels, y_pred)
        t_acc = accuracy_score(y_labels, y_pred)
        t_f1s = f1_score(y_labels, y_pred)
        # t_recall = recall_score(y_labels, y_pred)
        t_spe = tn / (tn + fp)  # pos_label=0,
        fpr, tpr, thresholds = roc_curve(y_labels, y_pred_prob)
        t_auc = roc_auc_score(y_labels, y_pred_prob)
        t_sen = tp / (tp + fn)
        t_pre = precision_score(y_labels, y_pred)

        self.performance_evaluation = [t_acc, acc_lower, acc_upper,
                                       t_f1s, f1s_lower, f1s_upper,
                                       t_sen, sen_lower, sen_upper,
                                       t_spe, spe_lower, spe_upper,
                                       t_auc, auc_lower, auc_upper,
                                       t_pre, pre_lower, pre_upper]
        plt.figure(figsize=(8, 6))
        for i, doc in enumerate(doctors):
            # 计算FPR = 1 - 特异性
            fpr_doctor = 1 - doc[2]
            tpr_doctor = doc[1]

            # plt.scatter(fpr_doctor, tpr_doctor,
            #             s=120, edgecolors='k', linewidths=1.5,
            #             marker=markers[i], color=colors[i],
            #             label='Radiologist %d (AUC = %0.3f (95%%CI = %0.3f - %0.3f))' % (
            #             i + 1, doc[0], doc[3], doc[4]))
            plt.scatter(fpr_doctor, tpr_doctor,
                        s=120, edgecolors='k', linewidths=1.5,
                        marker=markers[i], color=colors[i],
                        label='%s: AUC = %0.3f (95%%CI = %0.3f - %0.3f)' % (
                             doc[5], doc[0], doc[3], doc[4]))

            # AUC = XXX(95 % CI = xxx - xxx)
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label='LSF-Net: AUC = %0.3f (95%%CI = %0.3f - %0.3f)' % (t_auc, auc_lower, auc_upper))
        plt.plot([0, 1], [0, 1], color='#F5F5F5', lw=2, linestyle='--')  # 对角线
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.02])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(alpha=0.0)
        plt.show()
        return self.performance_evaluation

    def print_performance_evaluation(self):
        print(
            "      t_auc                t_acc                  t_f1_scor               t_sen                   t_spe                    t_prec")
        print("%.4f(%.4f,%.4f)  %.4f(%.4f,%.4f)  %.4f(%.4f,%.4f)  %.4f(%.4f,%.4f)  %.4f(%.4f,%.4f)  %.4f(%.4f,%.4f)" % (
            self.performance_evaluation[0], self.performance_evaluation[1], self.performance_evaluation[2],
            self.performance_evaluation[3], self.performance_evaluation[4], self.performance_evaluation[5],
            self.performance_evaluation[6], self.performance_evaluation[7], self.performance_evaluation[8],
            self.performance_evaluation[9], self.performance_evaluation[10], self.performance_evaluation[11],
            self.performance_evaluation[12], self.performance_evaluation[13], self.performance_evaluation[14],
            self.performance_evaluation[15], self.performance_evaluation[16], self.performance_evaluation[17]))
        return

    def print_performance_evaluation_new(self):
        print(
            "      t_auc                t_acc                  t_f1_scor               t_sen                   t_spe                    t_prec")
        print("%.4f(%.4f,%.4f)  %.4f(%.4f,%.4f)  %.4f(%.4f,%.4f)  %.4f(%.4f,%.4f)  %.4f(%.4f,%.4f)  %.4f(%.4f,%.4f)" % (
            self.performance_evaluation["AUC"], self.performance_evaluation["AUC_L"], self.performance_evaluation["AUC_U"],
            self.performance_evaluation["ACC"], self.performance_evaluation["ACC_L"], self.performance_evaluation["ACC_U"],
            self.performance_evaluation["F1S"], self.performance_evaluation["F1S_L"], self.performance_evaluation["F1S_U"],
            self.performance_evaluation["SEN"], self.performance_evaluation["SEN_L"], self.performance_evaluation["SEN_U"],
            self.performance_evaluation["SPE"], self.performance_evaluation["SPE_L"], self.performance_evaluation["SPE_U"],
            self.performance_evaluation["PRE"], self.performance_evaluation["PRE_L"], self.performance_evaluation["PRE_U"]))
        return

    def printf_confusion_matrix(self):
        print(self.confusion_matrix)

    def draw_confusion_matrix(self):
        plt.figure(figsize=(6, 4))
        self.confusion_matrix = np.transpose(self.confusion_matrix)
        self.confusion_matrix = self.confusion_matrix / np.sum(self.confusion_matrix, axis=0)
        self.confusion_matrix = np.transpose(self.confusion_matrix)
        sns.heatmap(self.confusion_matrix, annot=True, fmt='.1%', cmap='Blues', cbar=True,
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'],
                    )
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.show()
