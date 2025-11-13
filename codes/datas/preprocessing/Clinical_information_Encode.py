import sys

from codes.utils.r_common import *

if __name__ == '__main__':
    csv_path = r"../../../datasets/001.csv"
    mat_path = r"D:\Dataset\key_frame_10_all_mat_5_0\data"
    mat_save_path = r"D:\Dataset\key_frame_10_all_mat_5_c\data"
    if not os.path.exists(mat_save_path):
        os.makedirs(mat_save_path)
    datas = data_read_csv(csv_path)
    # print(len(datas))
    mat_list = os.listdir(mat_path)
    # print(mat_list)
    csv_list = []
    sex_list = []
    age_list = []
    pos_list = []
    are_list = []
    zjm_list = []
    hjm_list = []
    for i in range(len(datas) - 1):
        j = i + 1
        file = datas[j][0] + ".mat"
        csv_list.append(file)
        sex_list.append(datas[j][1])
        age_list.append(int(datas[j][2]) / 100)
        pos_list.append(int(datas[j][3]))
        are_list.append(int(datas[j][4]))
        zjm_list.append(float(datas[j][5]))
        hjm_list.append(float(datas[j][6]))
        if file not in mat_list:
            print(file + ": file is no exist")
    print("***************************************************")
    for i in range(len(mat_list)):
        if mat_list[i] not in csv_list:
            print(mat_list[i] + ": file is no exist")
    hzjm_list = zjm_list + hjm_list
    max_jm = max(hzjm_list)
    min_jm = min(hzjm_list)
    print(len(hzjm_list), max_jm, min_jm)
    max_pos = max(pos_list) + 1
    max_are = max(are_list) + 1
    max_age = max(age_list)
    if max_age < 100:
        max_age = 100
    print("max_pos_are:", max_pos, max_are)
    for i in range(len(mat_list)):
        mat_file_path = os.path.join(mat_path, mat_list[i])
        mat_save_dir = os.path.join(mat_save_path, mat_list[i])
        idx = csv_list.index(mat_list[i])
        # print(idx)
        # print(mat_file_path)
        # clinical_inf = csv_list[idx] + "," + sex_list[idx] + "," + age_list[idx] + "," + pos_list[idx] + "," + are_list[
        #     idx] + "," + zjm_list[idx] + "," + hjm_list[idx]
        # print(clinical_inf)
        sex_code = [1., 0.]
        if sex_list[idx] == '男':
            sex_code = [1., 0.]
        elif sex_list[idx] == '女':
            sex_code = [0., 1.]
        else:
            print("sex is error", csv_list[idx], sex_list[idx], idx)

        print(age_list[idx], pos_list[idx], are_list[idx], (zjm_list[idx] - min_jm) / (max_jm - min_jm),
              (hjm_list[idx] - min_jm) / (max_jm - min_jm))
        pos_code = [0.0 for _ in range(max_pos)]
        pos_code[pos_list[idx]] = 1.0
        are_code = [0.0 for _ in range(max_are)]
        are_code[are_list[idx]] = 1.0
        clinical_inf = sex_code
        clinical_inf.append(age_list[idx])
        clinical_inf += pos_code
        clinical_inf += are_code
        clinical_inf.append((zjm_list[idx] - min_jm) / (max_jm - min_jm))
        clinical_inf.append((hjm_list[idx] - min_jm) / (max_jm - min_jm))
        print(len(clinical_inf))
        # sys.exit()
        dat = io.loadmat(mat_file_path)
        label = dat['label']
        seg_label = dat['seg_label']
        all_dat = dat['ALL']
        print(mat_save_dir)
        io.savemat(mat_save_dir,
                   {'ALL': all_dat, 'label': label, 'seg_label': seg_label, 'clinical_inf': clinical_inf})
