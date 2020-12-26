import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from CustomDataset import CamvidDataset, denormalize
from models.UNet_Res import UNet
import configs as cfg
import pandas as pd
import numpy as np
import cv2 as cv

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 加载数据
test_data = CamvidDataset(cfg.test_path, cfg.test_label_path, cfg.json_path, mode="test")
test_db = DataLoader(test_data, batch_size=cfg.batch_size)

# 加载模型和参数
model = UNet(3, cfg.n_class)
model.load_state_dict(torch.load("C:/Users/WeiXing/Desktop/data_augmentation-segmentation_test/data_augmentation-segmentation_test/best.pth"))
net = model.eval()

# 标签预处理
pd_label_color = pd.read_csv(cfg.class_dict_path, sep=',')
name_value = pd_label_color['name'].values
num_class = len(name_value)
colormap = []
for i in range(num_class):
    tmp = pd_label_color.iloc[i]
    color = [tmp['r'], tmp['g'], tmp['b']]
    colormap.append(color)

cm = np.array(colormap).astype('uint8')

for i, (images, labels) in enumerate(test_db):
    images = images
    out = net(images)
    out = F.log_softmax(out, dim=1)
    pred_labels = out.max(1)[1]
    pred_labels = pred_labels.squeeze()
    pred_labels = pred_labels.cpu().data.numpy()
    preds = cm[pred_labels]
    labels = labels.numpy()
    images = denormalize(images.cpu(), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if cfg.batch_size != 1:
        for image, label, pred in zip(images, labels, preds):
            image = np.transpose(image.numpy(), (1, 2, 0)).astype(np.uint8)
            label = label.astype(np.uint8)
            label = cm[label]
            pred = pred.astype(np.uint8)
            cv.imshow("image", image)
            cv.imshow("label", label)
            cv.imshow("pred", pred)
            cv.waitKey()
    else:
        image = images.squeeze()
        label = np.squeeze(labels,axis=0)

        image = np.transpose(image.numpy(), (1, 2, 0)).astype(np.uint8)
        label = label.astype(np.uint8)
        label = cm[label]
        pred = preds.astype(np.uint8)
        cv.imshow("image", image)
        cv.imshow("label", label)
        cv.imshow("pred", pred)
        cv.waitKey()
        pass


    # pre1 = Image.fromarray(pre)
