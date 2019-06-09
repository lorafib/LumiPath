import random
import cv2
import numpy as np


class DualCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask=None):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class VerticalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            img = cv2.flip(img, 0)
            if mask is not None:
                mask = cv2.flip(mask, 0)
        return img, mask


class HorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            img = cv2.flip(img, 1)
            if mask is not None:
                mask = cv2.flip(mask, 1)
        return img, mask


class Resize:
    def __init__(self, h, w):
        self.h = h
        self.w = w

    def __call__(self, img, mask=None):
        img = cv2.resize(img, dsize=(self.w, self.h))
        if mask is not None:
            mask = cv2.resize(mask, dsize=(self.w, self.h))
        return img, mask


class Normalize:
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.mean = mean
        self.std = std

    def __call__(self, img, mask):
        img = img.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        return img, mask


class RandomBrightnessDual:
    def __init__(self, limit=0.1, prob=0.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            scale = np.random.uniform(low=-1.0, high=1.0)

            hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
            hsv = np.array(hsv, dtype=np.float64)
            hsv[:, :, 2] = hsv[:, :, 2] * (1.0 + self.limit * scale)
            hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
            mask = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2BGR)

            hsv_2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv_2 = np.array(hsv_2, dtype=np.float64)
            hsv_2[:, :, 2] = hsv_2[:, :, 2] * (1.0 + self.limit * scale)
            hsv_2[:, :, 2][hsv_2[:, :, 2] > 255] = 255
            img = cv2.cvtColor(np.array(hsv_2, dtype=np.uint8), cv2.COLOR_HSV2BGR)
        return img, mask


class RandomHueDual:
    def __init__(self, limit=0.3, prob=0.5):
        self.limit = limit
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            scale = np.random.uniform(low=-1.0, high=1.0)

            hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
            hsv = np.array(hsv, dtype=np.float64)
            hsv[:, :, 0] = hsv[:, :, 0] * (1.0 + self.limit * scale)
            hsv[:, :, 0][hsv[:, :, 0] > 179] = 179
            mask = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2BGR)

            hsv_2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv_2 = np.array(hsv_2, dtype=np.float64)
            hsv_2[:, :, 0] = hsv_2[:, :, 0] * (1.0 + self.limit * scale)
            hsv_2[:, :, 0][hsv_2[:, :, 0] > 179] = 179
            img = cv2.cvtColor(np.array(hsv_2, dtype=np.uint8), cv2.COLOR_HSV2BGR)

        return img, mask


class MaskLabel:
    def __call__(self, img, label):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray, 2, 1, 0)
        thresh = np.expand_dims(thresh, 2)
        thresh = np.concatenate((thresh, thresh, thresh), axis=2)
        label = label*thresh
        return img, label
