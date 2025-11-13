# coding:utf-8
# File:       r_file_class.py
# Author:     rifnyga
# Created on: 2024-03-18
# Function: 实现一些文件相关的功能

import os

class r_file_class:
    def __init__(self):
        self.m_filePath = ""
        self.m_files = []
        self.m_dirs = []
        self.m_self = []

    @staticmethod
    def r_join_path(dir_name, sub_dir_name):
        return os.path.join(dir_name, sub_dir_name)

    @staticmethod
    def r_mkdir(dir_name):
        return 1

    @staticmethod
    def r_is_exist_file(filename):
        if os.path.exists(filename):
            return 1
        else:
            return 0

    @staticmethod
    def r_is_exist_dir(dir_name):
        return 1

    def get_dir_list(self, path):
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isdir(file_path):
                self.m_dirs.append(file_path)
            else:
                self.m_files.append(file_path)
        return self.m_dirs

    def get_file_list(self, path):
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isdir(file_path):
                self.m_dirs.append(file_path)
            else:
                self.m_files.append(file_path)
        return self.m_files

    def rename_filename(self, idx, new_name):
        os.rename(self.m_files[idx], new_name)

    def rename_batch(self, subjoin_letter):
        for filename in self.m_files:
            new_filename = filename.replace('.', (subjoin_letter + '.'))
            os.rename(filename, new_filename)
