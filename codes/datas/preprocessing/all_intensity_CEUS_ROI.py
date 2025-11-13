# coding:utf-8
# File:       all_intensity_CEUS_ROI.py
# Author:     tfsy-rifnyga
# Created on: 2024-11-05
# function: 根据ROI区域完成关键帧的提取，主要针对造影


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

KEY_FRAMES_NUM = 10


def get_gray_key_frames(frames_list):

    hist_list = {}

    w = 1 # 连续取几张
    n = 9  # 一共取多少张

    print(len(frames_list))
    for i in range(len(frames_list)):
        img = frames_list[i]
        hist = cv2.calcHist([img], [0], None, [256], [0.0, 255.0])  # 256 * 1
        hist_list[i+1] = hist

    delta_f = {}
    key_number = []
    key_frames = []
    ordered_hist_list = sorted(hist_list)
    for i, img_number in enumerate(ordered_hist_list):
        if i < w:
            key_number.append(img_number)
            # key_frames.append(frames_list[ordered_hist_list[i]])
            continue
        if i + w >= len(hist_list.keys()):
            break
        sum_i = 0
        for neighborhood in range(-w, w + 1, 1):
            sum_i = sum_i + abs(hist_list[ordered_hist_list[i + neighborhood]] - hist_list[ordered_hist_list[i]]).sum()
        delta_f[img_number] = sum_i
    ordered_deltaf = sorted(delta_f.items(), key=lambda x: x[1], reverse=True)
    ordered_deltaf = [list(ech)[0] for ech in ordered_deltaf]
    key_number.extend(ordered_deltaf[:n])
    key_number = sorted(key_number)
    for i in range(len(key_number)):
        key_frames.append(frames_list[key_number[i]])
    print(key_number)
    # sys.exit()
    return key_frames, key_number


def key_frame_extern(m_image_list_path, m_mask_list_path, m_image_save_path):
    # m_image_list_path = r'D:\Dataset\test_image\US_IMG\1\LN ZHU SHI YU 578496720191211163308391'
    m_frames_list = []
    m_file_list = os.listdir(m_image_list_path)
    #file_list = []
    #for file_name in m_file_list:
    #    if file_name.split('.')[1]=='jpg':
    #        file_list.append(file_name)
    m_file_list.sort()
    m_file_list_len = len(m_file_list)
    print("file_list_len: ", m_file_list_len)
    print("正在加载所有帧图像.......")
    for file_name in m_file_list:
        image_file_path = os.path.join(m_image_list_path, file_name)
        mask_file_path = os.path.join(m_mask_list_path, file_name)
        print(image_file_path, mask_file_path)
        m_image = cv2.imread(image_file_path)
        m_mask = cv2.imread(mask_file_path)
        print("m_image.shape, m_mask.shape:", type(m_image), type(m_mask))
        img = m_image * m_mask
        m_gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        m_frames_list.append(copy.deepcopy(m_gray_image))
    print("加载所有帧图像完成.......")
    # print("计算所有相邻帧的pearson_correlation：")
    # m_correlation_list = []
    # for i in range(m_file_list_len - 1):
    #     curr_frame = m_frames_list[i]
    #     next_frame = m_frames_list[i + 1]
    #     m_correlation_list.append(calculate_pearson_correlation(curr_frame, next_frame))
    #     print("\r正在计算第%d和%d张之间的相关系数" % (i, i + 1), end='')
    # print("计算完成 关计算%d个相关系数，最大系数%f，最小系数%f" % (
    #     len(m_correlation_list), max(m_correlation_list), min(m_correlation_list)))
    # m_correlation_list_max = max(m_correlation_list)
    # m_correlation_list_min = min(m_correlation_list)
    # m_correlation_list_mean = sum(m_correlation_list) / len(m_correlation_list)
    # sys.exit()

    m_frames_idx = []
    m_key_frames = []
    # m_file_list_len
    print("开始提取关键帧.....")

    m_key_frames, m_frames_idx = get_gray_key_frames(m_frames_list)
    # while m_frames_idx[-1] / m_file_list_len < 0.2 or len(m_key_frames) < KEY_FRAMES_NUM:  # 15张图像
    #     m_key_frames, m_frames_idx = get_gray_key_frames(m_frames_list)
    #     print("提取帧数：", len(m_key_frames), "帧百分比：", m_frames_idx[-1] / m_file_list_len)
    #     if len(m_key_frames) > KEY_FRAMES_NUM:
    #         break
    m_key_frames_len = len(m_key_frames)
    print(m_key_frames_len)
    print("完成提取关键帧.....")

    for i in range(len(m_key_frames)):
        key_frame_file_name = os.path.join(m_image_save_path, m_file_list[m_frames_idx[i]])
        raw_frame_file_name = os.path.join(m_image_list_path, m_file_list[m_frames_idx[i]])
        raw_json_file_name = raw_frame_file_name.split('.')[0]+'.json'
        raw_json_file_name = str.replace(raw_json_file_name, 'raw_frame_ceus', 'raw_frame_ceus_new')
        key_json_file_name = key_frame_file_name.split('.')[0]+'.json'
        print("key_frame_file_name:", key_frame_file_name)
        print("raw_frame_file_name:", raw_frame_file_name)
        file_copy.copyfile(raw_frame_file_name, key_frame_file_name)
        file_copy.copyfile(raw_json_file_name, key_json_file_name)
    #


def extract_frames(source_dir, mask_dir, target_dir):
    print(">>>>source_dir>>>>:", source_dir)
    print(">>>>mask_dir>>>>:", mask_dir)
    print(">>>>target_dir>>>>:", target_dir)
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
        mask_class_dir = os.path.join(mask_dir, class_index)
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
            mask_image_frames_folder = os.path.join(mask_class_dir, video_prefix)
            if not os.path.exists(target_image_frames_folder):
                os.makedirs(target_image_frames_folder)
            print("source path:", source_image_frames_folder)
            print("mask path:", mask_image_frames_folder)
            print("target path:", target_image_frames_folder)
            print("///////////////////////////")
            key_frame_extern(source_image_frames_folder, mask_image_frames_folder, target_image_frames_folder)


if __name__ == '__main__':
    if __name__ == '__main__':
        parser = argparse.ArgumentParser(
            description="Extract frames of Kinetics400 dataset")
        parser.add_argument('--source_dir', type=str,
                            default=r'G:\key_frame_processing\raw_frame_ceus\lx\ceus',
                            help='the directory which is used to store the extracted frames')
        parser.add_argument('--mask_dir', type=str,
                            default=r'G:\key_frame_processing\raw_frame_ceus_mask\lx\ceus',
                            help='the directory which is used to store the extracted frames')
        parser.add_argument('--target_dir', type=str,
                            default=r'G:\key_frame_processing\raw_frame_ceus_key\lx\ceus',
                            help='the directory which is used to store the extracted frames')
        args = parser.parse_args()

        assert args.source_dir, "You must give the source_dir of raw videos!"
        assert args.mask_dir, "You must give the source_dir of raw videos!"
        assert args.target_dir, "You must give the traget_dir for storing the extracted frames!"

        import time

        tic = time.time()
        extract_frames(args.source_dir, args.mask_dir, args.target_dir)
        print(time.time() - tic)
