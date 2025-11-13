from codes.utils.r_common import *
import scipy.io as io
import string
import sys

if __name__ == '__main__':
    csv_dir = r"../../../datasets/恶性二分类.csv"  # 按该路径下csv文件加二分类标签
    data_path = r"D:\Dataset\key_frame_10_all_mat_5_c\data"  # 源数据路径，有临床信息
    t_data_path = r"D:\Dataset\key_frame_10_all_mat_5_c_2_ex_zj\data"  # 目标数据保存路径
    if not os.path.exists(t_data_path):
        os.makedirs(t_data_path)
    csv_data = data_read_csv(csv_dir)
    print(csv_data[1])
    # sys.exit()
    for i in range(len(csv_data) - 1):
        j = i + 1
        filename = csv_data[j][0] + ".mat"
        labels_2 = [0, 1]
        if csv_data[j][7] == '0':  # 转移癌
            labels_2 = [0, 1]
        elif csv_data[j][7] == '1':  # 淋巴瘤
            labels_2 = [1, 0]
        else:
            print("Data is error", j, filename)

            # labels_2 = csv_data[j][1] + ',' + csv_data[j][2]
        # labels_2 = labels_2.replace("\"", "")
        # labels_2 = [int(num) for num in labels_2.split(",")]

        print(filename, labels_2, labels_2[0], labels_2[1])
        mat_dir = os.path.join(data_path, filename)
        # t_filename = csv_data[j][0] + ".mat"    # "增生" [01]
        # t_mat_dir = os.path.join(t_data_path, t_filename)
        if labels_2[0] == 1:
            t_filename = csv_data[j][0] + "_1.mat"  # "增生" [01]
            t_mat_dir = os.path.join(t_data_path, t_filename)
        else:
            t_filename = csv_data[j][0] + "_0.mat"  # "结核"  [10]
            t_mat_dir = os.path.join(t_data_path, t_filename)
        print(mat_dir)
        print(t_mat_dir)
        data = io.loadmat(mat_dir)
        inputs = data["ALL"]
        labels = data['label']
        seg_labels = data['seg_label']
        clinical = data['clinical_inf']
        io.savemat(t_mat_dir, {'ALL': inputs, 'label': labels, 'seg_label': seg_labels, 'label_2': labels_2, 'clinical': clinical})
