# coding:utf-8
# File:       r_image_dataset_class.py
# Author:     rifnyga
# Created on: 2024-9-13

import os
import sys
import subprocess
import shutil
import argparse
import cv2


def extract_frames(source_dir, target_dir):
    print(source_dir)
    source_classes = os.listdir(source_dir)
    source_classes.sort()
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for class_index in source_classes:
        source_class_dir = os.path.join(source_dir, class_index)
        videos = os.listdir(source_class_dir)
        videos.sort()

        target_class_dir = os.path.join(target_dir, class_index)
        if not os.path.exists(target_class_dir):
            os.makedirs(target_class_dir)

        for each_video in videos:
            source_video_name = os.path.join(source_class_dir, each_video)
            video_prefix = each_video.split('.')[0]
            target_video_frames_folder = os.path.join(target_class_dir, video_prefix)
            if not os.path.exists(target_video_frames_folder):
                os.makedirs(target_video_frames_folder)
            target_frames = os.path.join(target_video_frames_folder, 'frame_%05d.jpg')

            # print(cv2.__version__)
            video = cv2.VideoCapture(source_video_name)  # 打开视频文件
            fps = video.get(cv2.CAP_PROP_FPS)  # 获取视频文件的帧速率
            frame_Count = video.get(cv2.CAP_PROP_FRAME_COUNT)  # 获取视频文件的帧数
            frame_Width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频文件的帧宽度
            frame_Height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频文件的帧高度
            print(fps, frame_Count, frame_Count/fps)
            try:
                # change videos to 30 fps and extract video frames
                subprocess.call(
                    'ffmpeg -nostats -loglevel 0 -i "%s" -filter:v fps=fps="%.2f" -s "%d"x"%d" -q:v 2 "%s"' %
                    (source_video_name, fps, frame_Width, frame_Height, target_frames), shell=True)

                # sanity check video frames
                video_frames = os.listdir(target_video_frames_folder)
                video_frames.sort()
            except:
                print('Video %s decode failed.' % (source_video_name))
                continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extract frames of Kinetics400 dataset")
    parser.add_argument('--source_dir', type=str,
                        default=r'G:\video_ue\良性\ue',
                        help='the directory which is used to store the extracted frames')
    parser.add_argument('--target_dir', type=str,
                        default=r'G:\video_ue\良性\ue_img',
                        help='the directory which is used to store the extracted frames')
    args = parser.parse_args()

    assert args.source_dir, "You must give the source_dir of raw videos!"
    assert args.target_dir, "You must give the traget_dir for storing the extracted frames!"

    import time

    tic = time.time()
    extract_frames(args.source_dir, args.target_dir)
    print(time.time() - tic)
