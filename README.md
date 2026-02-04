# LSF-Net
A Multimodal Ultrasound Foundation Model for Diagnosis of Unexplained Lymphadenopathy
---

数据集准备 (Dataset Preparation)

Due to medical data privacy concerns, please organize your private datasets according to the following directory structure. This project supports AVI video format.

```text
video_dataset/
├── bus_video
│   ├── malignant 3
│   │   ├── 1
│   │   │   └── patient 1
│   │   │       └── 1.avi
│   │   └── 2
│   ├── malignant
│   │   ├── 1
│   │   │   └── patient 1
│   │   │       └── 1.avi
│   │   └── 2
│   ├── malignant 1
│   │   ├── 1
│   │   │   └── patient 1
│   │   │       └── 1.avi
│   │   └── 2
│   ├── malignant 2
│   │   ├── 1
│   │   │   └── patient 1
│   │   │       └── 1.avi
│   │   └── 2
│   └── benign
│       ├── 1
│       │   └── patient 1
│       │       └── 1.avi
│       └── 2
├── cdfi_video
│   ├── malignant 3
│   ├── malignant
│   │   ├── 1
│   │   │   └── patient 1
│   │   └── 2
│   ├── malignant 1
│   ├── malignant 2
│   └── benign
├── ceus_video
│   ├── malignant 3
│   ├── malignant
│   │   ├── 1
│   │   │   └── patient 1
│   │   └── 2
│   ├── malignant 1
│   ├── malignant 2
│   └── benign
└── ue_video
    ├── malignant 3
    ├── malignant
    │   ├── 1
    │   │   └── patient 1
    │   └── 2
    ├── malignant 1
    ├── malignant 2
    └── benign

1. Environment Installation
conda create -n yourname python=3.10.13
conda activate yourname
2. Installation package
pip install -r requirements.txt

User Guide
Lymphadenopathy/
├── checkpoints
│   ├── foundation_model
│   │   └── vm_S_cnn_tf_T_mamba
│   └── reconversion
│       └── R_U_NetClass
├── codes
│   ├── datas
│   │   ├── classification_2_b
│   │   │   ├── add_labels_2_b.py
│   │   │   ├── r_dataset_load_mat.py
│   │   │   ├── r_dataset_load_mat_2.py
│   │   │   └── r_dataset_load_mat_test.py
│   │   ├── classification_2_bm
│   │   │   ├── add_labels_2_bm.py
│   │   │   ├── r_dataset_load_mat.py
│   │   │   └── r_dataset_load_mat_2.py
│   │   ├── classification_2_m
│   │   │   ├── add_labels_2_m.py
│   │   │   ├── r_dataset_load_mat.py
│   │   │   ├── r_dataset_load_mat_2.py
│   │   │   └── r_dataset_load_mat_test.py
│   │   ├── classification_4
│   │   │   ├── add_labels_4.py
│   │   │   ├── r_dataset_load_mat.py
│   │   │   └── r_dataset_load_mat_4.py
│   │   ├── foundation_model
│   │   │   └── r_dataset_load_mat.py
│   │   ├── preprocessing
│   │   │   ├── Clinical_information_Encode.py
│   │   │   ├── all_hog_lbp_CDFI.py
│   │   │   ├── all_hog_lbp_CEUS.py
│   │   │   ├── all_hog_lbp_UE.py
│   │   │   ├── all_hog_lbp_UE_ROI.py
│   │   │   ├── all_hog_lbp_US.py
│   │   │   ├── all_intensity_CEUS_ROI.py
│   │   │   ├── extern_frame.py
│   │   │   └── modality_4_in_1.py
│   │   └── reconversion
│   │       └── r_dataset_load_mat.py
│   ├── models
│   │   ├── classification_2_b
│   │   │   └── vm_classification_2_b_model.py
│   │   ├── classification_2_bm
│   │   │   ├── vm_classification_2_bm_model.py
│   │   │   └── vm_classification_2_bm_model_draw.py
│   │   ├── classification_2_m
│   │   │   └── vm_classification_2_m_model.py
│   │   ├── classification_4
│   │   │   ├── vm_classification_4_model.py
│   │   │   ├── vm_classification_4_model_2.py
│   │   │   └── vm_classification_4_model_draw.py
│   │   ├── foundation_model
│   │   │   ├── vm_S_cnn_T_lstm.py
│   │   │   ├── vm_S_cnn_T_mamba.py
│   │   │   ├── vm_S_cnn_T_tf.py
│   │   │   ├── vm_S_cnn_tf_T_lstm.py
│   │   │   ├── vm_S_cnn_tf_T_mamba.py
│   │   │   ├── vm_S_cnn_tf_T_tf.py
│   │   │   ├── vm_S_tf_T_lstm.py
│   │   │   ├── vm_S_tf_T_mamba.py
│   │   │   └── vm_S_tf_T_tf.py
│   │   ├── reconversion
│   │   │   ├── R_U_NetClass.py
│   │   │   ├── u_net.py
│   │   │   └── u_net_plus.py
│   │   └── utils
│   │       ├── vm_auto_router.py
│   │       └── vm_models_fusion.py
│   ├── models_train
│   │   ├── classification_2_b
│   │   │   └── classification_2_b_model_train.py
│   │   ├── classification_2_bm
│   │   │   └── classification_2_bm_model_train.py
│   │   ├── classification_2_m
│   │   │   └── classification_2_m_model_train.py
│   │   ├── classification_4
│   │   │   └── classification_4_model_train.py
│   │   ├── foundation_model
│   │   │   ├── foundation_model_train.py
│   │   │   └── mul_modality_mae_model_test.py
│   │   └── reconversion
│   │       ├── u_net_model_test.py
│   │       └── u_net_train.py
│   └── utils
│        ├── evaluation_function
│        │   └── classification_evaluation.py
│        ├── loss_function
│        │   ├── common_loss_funtion.py
│        │   └── r_simm_class.py
│        ├── csvClass.py
│        ├── log_save.py
│        ├── r_common.py
│        └── r_file_class.py
├── datasets
│   ├── p_value_calculate.csv
│   ├── p_value_ex.csv
│   ├── p_value_lx.csv
│   ├── 恶性二分类.csv
│   ├── 良性二分类.csv
│   └── 转移来源4分类.csv
├── doc
├── results
│   ├── foundation_model
│   │   └── vm_S_cnn_tf_T_mamba
│   │       ├── vm_S_cnn_tf_T_mamba_F_0.csv
│   │       ├── vm_S_cnn_tf_T_mamba_F_1.csv
│   │       ├── vm_S_cnn_tf_T_mamba_F_2.csv
│   │       ├── vm_S_cnn_tf_T_mamba_F_3.csv
│   │       └── vm_S_cnn_tf_T_mamba_F_4.csv
│   └── reconversion
│       └── R_U_NetClass
│           ├── R_U_NetClass_F_2.csv
│           ├── R_U_NetClass_F_3.csv
│           └── R_U_NetClass_F_4.csv
└── requirements.txt
