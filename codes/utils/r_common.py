# coding:utf-8
import random
import os
import torch
import csv
import codecs
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
from PIL import Image


def set_seed(seed=1):
    seed_value = seed  # the number of seed
    np.random.seed(seed_value)  # set random seed for numpy
    random.seed(seed_value)  # set random seed for python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # avoid hash random
    torch.manual_seed(seed_value)  # set random seed for CPU
    torch.cuda.manual_seed(seed_value)  # set random seed for one GPU
    torch.cuda.manual_seed_all(seed_value)  # set random seed for all GPU
    torch.backends.cudnn.deterministic = True  # set random seed for convolution


def inf_data(datas):
    data_max2 = 2 * np.max(datas)
    datas[datas == -np.inf] = data_max2
    datas_min = np.min(datas)
    datas -= datas_min
    datas[datas == data_max2 - datas_min] = 0
    return datas


def r_display_image(filename, data_name, threshold):
    image_data = io.loadmat(filename)[data_name]
    # print(np.max(image_data), np.min(image_data))
    # image_data = (image_data-np.min(image_data))/(np.max(image_data)-np.min(image_data))
    image_data[image_data >= -threshold] += threshold
    image_data[image_data < -threshold] = 0
    # print(np.max(image_data), np.min(image_data))
    plt.imshow(image_data, cmap="gray")
    plt.show()


def data_write_csv(file_name, data_list):  # file_name为写入CSV文件的路径，data_list为要写入数据列表
    file_csv = codecs.open(file_name, 'w+', 'utf-8')  # 追加
    writer = csv.writer(file_csv, delimiter=',', quotechar='\'', quoting=csv.QUOTE_MINIMAL)
    for dat in data_list:
        writer.writerow(dat)
    print("保存文件成功:", file_name)


def data_read_csv(file_name):
    data_list = []
    file_csv = codecs.open(file_name, 'r', 'utf-8')
    reader = csv.reader(file_csv, delimiter=',', quotechar='\'', quoting=csv.QUOTE_MINIMAL)
    for dat in reader:
        data_list.append(dat)
    print("读取文件成功:", file_name)
    return data_list


def tensor_expand(t_x, padding_size, value):
    new_row = t_x.size()[0] + 2 * padding_size[0]
    new_col = t_x.size()[1] + 2 * padding_size[1]
    if torch.cuda.is_available():
        y = torch.ones((new_row, new_col)).cuda() * value
    else:
        y = torch.ones((new_row, new_col)) * value
    y[padding_size[0]:t_x.size()[0] + padding_size[0], padding_size[1]:t_x.size()[1] + padding_size[1]] = t_x
    return y


class Rescale(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        img = np.resize(image, (2, self.output_size, self.output_size))
        lab = np.resize(label, (self.output_size, self.output_size))
        return {'image': img, 'label': lab}


class RandomCrop(object):
    """随机裁剪样本中的图像.

    Args:
       output_size（tuple或int）：所需的输出大小。 如果是int，方形裁剪是。
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[1:]
        # print(h, w)
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[:, top: top + new_h,
                left: left + new_w]

        label = label[top: top + new_h,
                left: left + new_w]

        return {'image': image, 'label': label}


class ToTensor(object):
    """将样本中的ndarrays转换为Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # 交换颜色轴因为
        # numpy包的图片是: H * W * C
        # torch包的图片是: C * H * W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label)}


class Rotate(object):
    def __call__(self, sample, angle):
        image, label = sample['image'], sample['label']
        image = np.clip(image, 0, 1)
        label = np.clip(label, 0, 1)
        image = (image * 255).astype(np.uint8)
        label = (label * 255).astype(np.uint8)
        image = image.transpose((1, 2, 0))
        # label = label.transpose((1, 2, 0))
        # print("image.shape:", image.shape)
        # print("label.shape:", label.shape)
        im = Image.fromarray(image)
        la = Image.fromarray(label[0])
        im = im.rotate(angle, Image.NEAREST, expand=False)
        la = la.rotate(angle, Image.NEAREST, expand=False)
        image = np.asarray(im)
        label[0] = np.asarray(la)
        image = image.transpose((2, 0, 1))
        # label = label.transpose((2, 0, 1))
        # print("image.shape:", image.shape)
        image = (image / 255).astype(np.float32)
        label = (label / 255).astype(np.float32)
        return {'image': image, 'label': label}


def mask_code_generator(list_len, num_prob_list):
    t_mask_list = [1 for _ in range(list_len)]
    for i in range(list_len):
        t_mask_list[i] = np.random.choice([0, 1], size=1, p=[(1-num_prob_list[i]), num_prob_list[i]])[0]
    # print(t_mask_list)
    return t_mask_list
