# coding:utf-8
# File:       lbp_test.py
# Author:     tfsy-rifnyga
# Created on: 2024-09-20

import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'
import sys

sys.path.append(os.path.abspath(os.path.join("./codes", "..")))
import argparse
import copy
import cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure
from scipy.stats import pearsonr
from codes.utils.r_file_class import *
import matplotlib
from skimage.feature import local_binary_pattern

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import math
import shutil as file_copy

KEY_FRAMES_NUM = 12


def calculate_pearson_correlation_hl(frame1, frame2):
    # 将图像转换为灰度
    gray1_hog = get_hog_vector(frame1)
    gray2_hog = get_hog_vector(frame2)
    gray1_lbp = get_lbp_vector(frame1)
    gray2_lbp = get_lbp_vector(frame2)
    # 将图像展平为一维数组
    # print(type(gray1_hog), type(gray2_lbp), gray1_hog.shape, gray2_lbp.shape)

    gray1_l = np.array(gray1_lbp)
    gray1_h = np.array(gray1_hog)
    gray1 = np.append(gray1_h, gray1_l)
    # gray1 = gray1_h
    gray2_l = np.array(gray2_lbp)
    gray2_h = np.array(gray2_hog)
    gray2 = np.append(gray2_h, gray2_l)
    # gray2 = gray2_h

    gray1_flat = gray1.flatten()
    gray2_flat = gray2.flatten()

    # 计算皮尔逊相关系数
    correlation, _ = pearsonr(gray1_flat, gray2_flat)
    return correlation


def calculate_pearson_correlation(frame1, frame2):
    # 将图像转换为灰度
    gray1_hog = get_hog_vector(frame1)
    gray2_hog = get_hog_vector(frame2)
    # gray1_lbp = get_lbp_vector(frame1)
    # gray2_lbp = get_lbp_vector(frame2)
    # 将图像展平为一维数组
    # print(type(gray1_hog), type(gray2_lbp), gray1_hog.shape, gray2_lbp.shape)

    # gray1_l = np.array(gray1_lbp)
    gray1_h = np.array(gray1_hog)
    # gray1 = np.append(gray1_h, gray1_l)
    gray1 = gray1_h
    # gray2_l = np.array(gray2_lbp)
    gray2_h = np.array(gray2_hog)
    # gray2 = np.append(gray2_h, gray2_l)
    gray2 = gray2_h

    gray1_flat = gray1.flatten()
    gray2_flat = gray2.flatten()

    # 计算皮尔逊相关系数
    correlation, _ = pearsonr(gray1_flat, gray2_flat)
    return correlation


def get_lbp_vector(frame):
    radius = 5  # 邻域半径
    n_points = 32 * radius  # 圆形邻域中的点数
    # 计算 LBP 特征图
    lbp = local_binary_pattern(frame, n_points, radius, method='uniform')

    # 统计 LBP 特征图的直方图
    # 生成直方图的bin数，通常在uniform模式下有 P + 2 个不同的值
    n_bins = int(lbp.max() + 1)  # 确定bin数
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))

    # 归一化直方图，作为特征向量
    hist = hist.astype('float')
    hist /= (hist.sum() + 1e-6)  # 避免除零

    # 输出特征向量
    # print("LBP 特征向量：", hist)
    return hist


def normal_fun_1(x_v, sgm, u):
    return (1 / (math.sqrt(2 * math.pi) * sgm)) * math.exp(-((x_v - u) * (x_v - u)) / (2 * sgm * sgm))


def normal_fun(x_v, sgm, u):
    return normal_fun_1(x_v, sgm, 0) / normal_fun_1(0, sgm, 0)


def adaptive_key_frame_selection(frames_list, threshold_max=1, threshold_mean=1, sgm=1.25,
                                 bias=0.8):  # US : 0.38  UE: 0.40
    x = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  # 正态分布的X取值
    # print(x)
    j = 0
    is_ok = 0
    max_v = normal_fun_1(0.0, sgm, 0)
    if len(frames_list) > KEY_FRAMES_NUM:  # 当实际帧数大于要取的关键帧数时
        prev_frame = frames_list[0]# 初始化关键帧列表//
        key_frames = [prev_frame]  # 将第一帧作为关键帧
        frames_idx = [1]
        is_ok = 0
        for i in range(len(frames_list) - 1):
            # 计算当前帧与前一关键帧之间的皮尔逊相关系数
            current_frame = frames_list[i + 1]
            correlation = calculate_pearson_correlation(prev_frame, current_frame)
            threshold = normal_fun_1(x[j], sgm, 0) / max_v * threshold_max + bias  # 0.92 最大相似度
            # print(threshold, correlation)
            if correlation < threshold:
                key_frames.append(current_frame)
                frames_idx.append(i + 1)
                prev_frame = current_frame
                j += 1
                if j > KEY_FRAMES_NUM:  # 15张图像
                    break        
    else:
        key_frames = [] 
        frames_idx = [] 
        is_ok = 1
        for i in range(len(frames_list)):
            key_frames.append(frames_list[i])
            frames_idx.append(i+1)       
    return key_frames, frames_idx, is_ok


def get_hog_vector(image):
    # image = cv2.imread(image_path)
    gray_image = image
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 计算HOG特征
    # 参数说明：
    # orientations=9 表示将360度分成9份，即每个bin覆盖40度
    # pixels_per_cell=(8, 8) 表示每个cell的大小为8x8像素
    # cells_per_block=(2, 2) 表示每个block由2x2个cell组成
    # block_norm='L2' 表示使用L2范数进行归一化
    # channel_axis=None 表示图像是灰度的，没有通道轴
    hog_features = hog(gray_image,
                       orientations=9,
                       pixels_per_cell=(64, 64),
                       cells_per_block=(2, 2),
                       block_norm='L2',
                       channel_axis=None)

    # 调整HOG特征的对比度，使其范围在[0, 1]之间
    hog_features = exposure.rescale_intensity(hog_features, in_range=(0, 1))

    # 将HOG特征展平为一维数组
    hog_features = hog_features.flatten()
    return hog_features


def key_frame_extern(m_image_list_path, m_image_save_path):
    # m_image_list_path = r'D:\Dataset\test_image\US_IMG\1\LN ZHU SHI YU 578496720191211163308391'
    m_frames_list = []
    m_file_obj = r_file_class()
    m_file_list = m_file_obj.get_file_list(m_image_list_path)
    m_file_list_len = len(m_file_list)
    print("file_list_len: ", m_file_list_len)
    print("正在加载所有帧图像.......")
    for i in range(len(m_file_list)):
        m_image = cv2.imread(m_file_list[i])
        m_gray_image = cv2.cvtColor(m_image, cv2.COLOR_BGR2GRAY)
        m_frames_list.append(copy.deepcopy(m_gray_image))
    print("加载所有帧图像完成.......")
    print("计算所有相邻帧的pearson_correlation：")
    m_correlation_list = []
    for i in range(m_file_list_len - 1):
        curr_frame = m_frames_list[i]
        next_frame = m_frames_list[i + 1]
        m_correlation_list.append(calculate_pearson_correlation(curr_frame, next_frame))
        print("\r正在计算第%d和%d张之间的相关系数" % (i, i + 1), end='')

    m_correlation_list_max = max(m_correlation_list)
    m_correlation_list_min = min(m_correlation_list)
    m_correlation_list_mean = sum(m_correlation_list) / len(m_correlation_list)
    print("计算完成 关计算%d个相关系数，最大系数%f，最小系数%f，平均系数%f" % (
        len(m_correlation_list), m_correlation_list_max, m_correlation_list_min, m_correlation_list_mean))
    # m_correlation_list_mean越小说明视频帧与帧之间变化越明显，否则说明视频帧与帧之间变化不明显
    # sys.exit()

    m_frames_idx = []
    m_key_frames = []
    m_sgm = m_correlation_list_mean
    m_bias = 0.0
    print("开始提取关键帧.....")
    j_count = 0
    d_bias = m_correlation_list_min
    key1 = 0
    while True:  # 15张图像
        m_key_frames, m_frames_idx, is_ok = adaptive_key_frame_selection(m_frames_list, m_correlation_list_max,
                                                                         m_correlation_list_mean, sgm=m_sgm,
                                                                         bias=m_bias)
        print(j_count, "m_bias：", m_bias, "提取帧数：", len(m_key_frames), "帧百分比：",
              m_frames_idx[-1] / m_file_list_len)
        if is_ok == 1:  # 原始视频帧小于关键帧个数时
            break
        if m_frames_idx[-1] / m_file_list_len > 0.9 and len(m_key_frames) >= KEY_FRAMES_NUM:
            # 当所得关键帧个数大于等于关键帧个数，且所取最后一帧处于帧进度的90%以外
            break
            
        # 跳出策略
        j_count += 1
        if j_count > 60:
            break
        elif j_count > 40:
            if len(m_key_frames) > KEY_FRAMES_NUM and m_frames_idx[-1] / m_file_list_len > 0.7:
                break
        elif j_count > 20:
            if len(m_key_frames) > KEY_FRAMES_NUM and m_frames_idx[-1] / m_file_list_len > 0.8:
                break

        # bias的调整策略，折半法查找bias
        if m_frames_idx[-1] / m_file_list_len < 0.9 and len(m_key_frames) > KEY_FRAMES_NUM:
            m_bias -= d_bias
            if key1 == 1:
                d_bias /= 2
            key1 = 0
        elif m_frames_idx[-1] / m_file_list_len < 0.9 and len(m_key_frames) < KEY_FRAMES_NUM:
            m_bias += d_bias
            if key1 == 0:
                d_bias /= 2
            key1 = 1
        else:
            if key1 == 1:
                m_bias += d_bias
            else:
                m_bias -= d_bias
        if math.fabs(d_bias) < 1e-6:
            d_bias *= 100

    m_key_frames_len = len(m_key_frames)
    print(m_key_frames_len)
    print("完成提取关键帧.....")
    # row_num = (m_key_frames_len - 1) // 5 + 1
    # col_num = 5
    for i in range(len(m_key_frames)):
        #     plt.subplot(row_num, col_num, i + 1)
        #     plt.imshow(m_key_frames[i], cmap="gray")
        key_frame_file_name = os.path.join(m_image_save_path, "frame_%05d.jpg" % (m_frames_idx[i]+1))
        print(key_frame_file_name, m_file_list[m_frames_idx[i]])
        file_copy.copyfile(m_file_list[m_frames_idx[i]], key_frame_file_name)
    #     plt.title("frames[%d]" % m_frames_idx[i])
    #     plt.axis('off')
    # fig_file_name = os.path.join(m_image_save_path, "0.jpg")
    # plt.savefig(fig_file_name)
    # plt.show()


def extract_frames(source_dir, target_dir):
    print(source_dir)
    source_classes = os.listdir(source_dir)
    source_classes.sort()
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    count = 0
    for class_index in source_classes:
        if not count % 10:
            print("Current class inx: ", count)
        count += 1
        source_class_dir = os.path.join(source_dir, class_index)
        videos = os.listdir(source_class_dir)
        videos.sort()

        target_class_dir = os.path.join(target_dir, class_index)
        if not os.path.exists(target_class_dir):
            os.makedirs(target_class_dir)

        for each_video in videos:
            source_video_name = os.path.join(source_class_dir, each_video)
            video_prefix = each_video.split('.')[0]
            # print("video_prefix:" , video_prefix)
            target_image_frames_folder = os.path.join(target_class_dir, video_prefix)
            source_image_frames_folder = os.path.join(source_class_dir, video_prefix)
            if not os.path.exists(target_image_frames_folder):
                os.makedirs(target_image_frames_folder)
            print("source path:", source_image_frames_folder)
            print("target path:", target_image_frames_folder)
            print("///////////////////////////")
            key_frame_extern(source_image_frames_folder, target_image_frames_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract frames of Kinetics400 dataset")
    parser.add_argument('--source_dir', type=str,
                        default=r'D:\Dataset\key_frame_test\US_IMG',
                        help='the directory which is used to store the extracted frames')
    parser.add_argument('--target_dir', type=str,
                        default=r'D:\Dataset\key_frame_test\US_KEY',
                        help='the directory which is used to store the extracted frames')
    args = parser.parse_args()

    assert args.source_dir, "You must give the source_dir of raw videos!"
    assert args.target_dir, "You must give the traget_dir for storing the extracted frames!"

    import time

    tic = time.time()
    extract_frames(args.source_dir, args.target_dir)
    print(time.time() - tic)
