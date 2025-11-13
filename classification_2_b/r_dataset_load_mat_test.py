import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from codes.datas.ablation_classification_2.r_dataset_load_mat import *
from torch.utils.data import DataLoader


if __name__ == '__main__':
    datasets = RDatasetFiveFoldOneModelityClass()
    test_datasets = RDatasetFiveFoldOneModelityClass()
    datasets.set_modality_name("ALL")
    test_datasets.set_modality_name("ALL")
    datasets_path = r"D:\Dataset\key_frame_10_all_mat_5_c_2_lex_a"
    datasets.create_dataset(datasets_path, "train_0.txt")
    print(datasets.get_dataset_count())
    train_data_loader = DataLoader(datasets, batch_size=1, shuffle=True, num_workers=1)
    for i, data in enumerate(train_data_loader, 0):
        inputs = data['inputs']  # us_inputs,ue_inputs,cd_inputs,ce_inputs
        labels = data['labels']
        # clinical = data['clinical']
        filename = data['filename']
        print(filename, labels[0])
