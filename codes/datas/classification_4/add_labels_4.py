from codes.utils.r_common import *
import scipy.io as io
import string

if __name__ == '__main__':
    csv_dir = r"../../../datasets/转移来源4分类.csv"
    data_path = r"D:\Dataset\key_frame_10_all_mat_5_c\data"
    t_data_path = r"D:\Dataset\key_frame_10_all_mat_5_c_4_zy\data"  # 目标数据保存路径
    if not os.path.exists(t_data_path):
        os.makedirs(t_data_path)
    csv_data = data_read_csv(csv_dir)
    print(csv_data[1])
    for i in range(len(csv_data) - 1):
        j = i + 1
        filename = csv_data[j][0] + ".mat"
        labels_4 = [1, 0, 0, 0]
        if csv_data[j][7] == '0':  # 肺
            labels_4 = [1, 0, 0, 0]
        elif csv_data[j][7] == '1':  # 上呼吸道
            labels_4 = [0, 1, 0, 0]
        elif csv_data[j][7] == '2':  # 甲乳
            labels_4 = [0, 0, 1, 0]
        elif csv_data[j][7] == '3':  # 消化道
            labels_4 = [0, 0, 0, 1]
        else:
            print("Data is error", j, filename)
        # labels_2 = csv_data[j][1] + ',' + csv_data[j][2] + ',' + csv_data[j][3] + ',' + csv_data[j][4]
        # labels_2 = labels_2.replace("\"", "")
        # labels_2 = [int(num) for num in labels_2.split(",")]

        print(filename, labels_4, labels_4[0], labels_4[1], labels_4[2], labels_4[3])
        mat_dir = os.path.join(data_path, filename)
        if labels_4[3] == 1:
            t_filename = csv_data[j][0] + "_3.mat"  # 消化道   [1000]
        elif labels_4[2] == 1:
            t_filename = csv_data[j][0] + "_2.mat"  # 甲乳     [0100]
        elif labels_4[1] == 1:
            t_filename = csv_data[j][0] + "_1.mat"  # 上呼吸道  [0010]
        elif labels_4[0] == 1:
            t_filename = csv_data[j][0] + "_0.mat"  # 肺     [0001]
        else:
            print("Data is error", j, filename)
        t_mat_dir = os.path.join(t_data_path, t_filename)
        print(mat_dir)
        print(t_mat_dir)
        data = io.loadmat(mat_dir)
        inputs = data["ALL"]
        labels = data['label']
        seg_labels = data['seg_label']
        clinical = data['clinical_inf']
        io.savemat(t_mat_dir,
                   {'ALL': inputs, 'label': labels, 'seg_label': seg_labels, 'label_4': labels_4, 'clinical': clinical})
