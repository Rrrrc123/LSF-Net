# coding:utf-8
# File:       modality_4_in_1.py
# Author:     tfsy-rifnyga
# Created on: 2024-11-2

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import scipy.io as io

if __name__ == '__main__':
    us_path = r"D:\Dataset\key_frame_10_us_mat_5_0\data"
    ue_path = r"D:\Dataset\key_frame_10_ue_mat_5_0\data"
    cd_path = r"D:\Dataset\key_frame_10_cdfi_mat_5_0\data"
    ce_path = r"D:\Dataset\key_frame_10_ceus_mat_5_0\data"
    all_path = r"D:\Dataset\key_frame_10_all_mat_5_0\data"
    if not os.path.exists(all_path):
        os.makedirs(all_path)
    us_file_list = os.listdir(us_path)
    ue_file_list = os.listdir(ue_path)
    cd_file_list = os.listdir(cd_path)
    ce_file_list = os.listdir(ce_path)
    for i in range(len(us_file_list)):
        us_file_name = os.path.join(us_path, us_file_list[i])
        ue_file_name = os.path.join(ue_path, ue_file_list[i])
        cd_file_name = os.path.join(cd_path, cd_file_list[i])
        ce_file_name = os.path.join(ce_path, ce_file_list[i])
        all_file_name = os.path.join(all_path, us_file_list[i])
        us_data = io.loadmat(us_file_name)
        ue_data = io.loadmat(ue_file_name)
        cd_data = io.loadmat(cd_file_name)
        ce_data = io.loadmat(ce_file_name)
        us_inputs = us_data["US"]
        ue_inputs = ue_data["UE"]
        cd_inputs = cd_data["CDFI"]
        ce_inputs = ce_data["CEUS"]
        us_labels = us_data["label"]
        ue_labels = ue_data["label"]
        cd_labels = cd_data["label"]
        ce_labels = ce_data["label"]
        print(us_file_name, len(us_inputs), us_inputs[0].shape, us_labels)
        print(ue_file_name, len(ue_inputs), ue_inputs[0].shape, ue_labels)
        print(cd_file_name, len(cd_inputs), cd_inputs[0].shape, cd_labels)
        print(ce_file_name, len(ce_inputs), ce_inputs[0].shape, ce_labels)
        print("**************************************************************")
        inputs = np.concatenate([us_inputs,ue_inputs,cd_inputs,ce_inputs], axis=0)

        print(inputs.shape)
        io.savemat(all_file_name, {'ALL': inputs, 'label': us_labels})

