import random
import torch
import numpy as np

from torchvision.transforms import functional as F


class Compose(object):
    """组合多个transform函数"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    """将PIL图像转为Tensor"""

    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip(object):
    """随机水平翻转图像以及bboxes"""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)  # 水平翻转图片
            bbox = target["boxes"]
            # bbox: xmin, ymin, xmax, ymax
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]  # 翻转对应bbox坐标信息
            target["boxes"] = bbox
        return image, target


class RandomVerticalFlip(object):
    # 随机垂直翻转
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image[::-1, :, :]
            bbox = target["boxes"]
            bbox[:, [1, 3]] = height - bbox[:, [3, 1]]
            target["boxes"] = bbox
        return image, target


class RandomRotate90(object):
    # 随机旋转90度
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = np.rot90(image, 1, (0, 1))
            bbox = target["boxes"]
            x = bbox[:, 0].copy()
            y = bbox[:, 1].copy()
            w = (bbox[:, 2] - bbox[:, 0]).copy()
            h = (bbox[:, 3] - bbox[:, 1]).copy()
            bbox[:, 0] = (width - height) // 2 + y
            bbox[:, 1] = (width + height) // 2 - x - w
            bbox[:, 2] = bbox[:, 0] + h
            bbox[:, 3] = bbox[:, 1] + w
            target["boxes"] = bbox
        return image, target
