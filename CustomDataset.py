import torch
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np
import os
import cv2 as cv
from uilts.parse_cfg import parse_json
import configs as cfg
from pipelines.compose import Compose
import pandas as pd


class LabelProcessor:
    '''标签预处理'''
    def __init__(self, file_path):
        colormap = self.read_color_map(file_path)

        assert len(colormap)==cfg.n_class

        # 对标签做编码，返回哈希表
        self.cm2lbl = self.encode_label_pix(colormap)

    # 将mask中的RGB转成编码的label
    def encode_label_img(self, img):
        data = np.array(img, np.int32)
        idx = (data[:, :, 0] * 256+data[:, :, 1])*256 + data[:, :, 2]

        # print('index', self.cm2lbl[128*256*256])
        # 返回编码后的label
        return np.array(self.cm2lbl[idx], np.int64)

    # 返回一个哈希映射  再 3维256 空间中
    @staticmethod
    def encode_label_pix(colormap):
        cm2lbl = np.zeros(256**3)  # 3维的256的空间 打成一维度
        for i, cm in enumerate(colormap):
            cm2lbl[(cm[0]*256+cm[1])*256+cm[2]] = i
        return cm2lbl

    # 读取csv文件
    @staticmethod
    def read_color_map(file_path):
        pd_label_color = pd.read_csv(file_path, sep=',')
        colormap = []

        for i in range(len(pd_label_color.index)):
            tmp = pd_label_color.iloc[i]
            color = [tmp['r'], tmp['g'], tmp['b']]
            colormap.append(color)

        return colormap


class CamvidDataset(Dataset):

    def __init__(self, img_path, label_path, json_path, mode="train"):
        self.imgs = self.read_file(img_path)
        self.labels = self.read_file(label_path)
        # print(len(self.imgs), len(self.labels))
        assert len(self.imgs) == len(self.labels), "label 和 image 数据长度不同"

        config = parse_json(json_path)
        if mode == 'train':
            self.train_pipeline = Compose(config['train'])
        else:
            self.train_pipeline = Compose(config['test'])

        self.tf = transforms.Compose([
            lambda x:torch.tensor(x, dtype=torch.float32)])



    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        img = self.imgs[item]
        label = self.labels[item]

        image = cv.imread(img)
        label = cv.imread(label)[..., ::-1]

        # cv.imshow("image", image)
        # cv.imshow("label", label)
        # cv.waitKey()
        img, label = self.img_transform(image, label)

        return img, label

    def read_file(self, path):
        '''从文件夹中读取数据'''
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, file) for file in files_list]
        file_path_list.sort()
        return file_path_list

    def img_transform(self, image, mask):
        '''图像数据预处理并转成tensor格式'''

        # 获取图像信息
        data = {"type": "segmentation"}
        data["image"] = image
        data["mask"] = mask

        # 数据增强
        augment_result = self.train_pipeline(data)

        image = augment_result["image"]
        mask = augment_result["mask"]

        # 转成tensor格式
        image = self.tf(np.transpose(image, (2, 0, 1)))

        # 对标签进行编码，转成tensor
        mask = label_processor.encode_label_img(mask)
        mask = torch.from_numpy(mask)

        return image, mask


label_processor = LabelProcessor(cfg.class_dict_path)


def denormalize(x_hat, mean=[0.2826372, 0.2826372, 0.2826372], std=[0.30690703, 0.30690703, 0.30690703]):

    mean = torch.tensor(mean).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
    std = torch.tensor(std).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
    x = x_hat * std + mean
    return x*255


if __name__ == "__main__":
    test = CamvidDataset(cfg.train_path, cfg.train_label_path, cfg.json_path, mode="train")
    from torch.utils.data import DataLoader

    test_db = DataLoader(test, batch_size=cfg.batch_size)
    for img, label in test_db:
        images = denormalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        images = images.numpy()
        labels = label.numpy()
        for image, label in zip(images, labels):
            image = np.transpose(image, (1, 2, 0))
            cv.imshow("img", image.astype(np.uint8))
            cv.imshow("label", label.astype(np.uint8))
            cv.waitKey()


