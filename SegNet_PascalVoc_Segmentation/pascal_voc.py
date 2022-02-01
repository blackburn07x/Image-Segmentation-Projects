import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import style
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Preprocess data
images_path = "ImageSets\\Segmentation\\train.txt"
train_label_path = "SegmentationClass\\"
train_image_path = "JPEGImages\\"

HEIGHT = 128
WIDTH = 128
CHANNELS = 3
CLASSES = 21

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
               'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

def get_images(path):
    with open(path, 'r') as f:
        images = f.read().split()
    return images

def read_image(img):
    img = cv2.imread(img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (HEIGHT, WIDTH))
    return img

def preprocess_onehotmask(mask):
    output_mask = np.zeros((HEIGHT, WIDTH, CLASSES), dtype=np.uint8)
    for i, color_code in enumerate(VOC_COLORMAP):
        output_mask[:, :, i] = np.all(np.equal(mask, color_code), axis=-1).astype(float)
    return output_mask

def read_dataset():
    images = get_images(images_path)
    CLASSES = 21

    imgs = np.empty((len(images), HEIGHT, WIDTH, CHANNELS), dtype=np.uint8)
    masks = np.empty((len(images), HEIGHT, WIDTH, CLASSES), dtype=np.uint8)

    for i in tqdm(range(len(images))):
        imgs[i, :, :, :] = read_image(train_image_path + images[i] + ".jpg")
        masks[i, :, :, :] = preprocess_onehotmask(read_image(train_label_path + images[i] + ".png"))
    return imgs, masks


