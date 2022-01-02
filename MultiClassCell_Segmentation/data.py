import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

#read_dataset
#prepare images
images = os.listdir("train/")
#read train.csv
train = pd.read_csv("train.csv")
temp = train.drop_duplicates(subset = ["id"])

def rle_mask(rle,h,w):
    """
    :param rle: run_length encoding of mask
    :param h: height
    :param w: weight
    :return:
    Image mask of shape (h,w,1)
    """
    img = np.zeros((h*w,1),dtype=np.uint8)
    enc = rle.split()
    ffrom = np.array([x for x in enc[0:][::2]],dtype=int)
    to = np.array([x for x in enc[1:][::2]],dtype=int)
    ends = ffrom + to
    for f,t in zip(ffrom,ends):
        img[f:t] = 1
    img = img.reshape((h,w,1))
    return img

def read_masks(temp,h,w):
    """
    :param temp:No duplicate dataset
    :param h: height
    :param w: width
    :return:
    masks of shape (606,520,704,1)
    """
    keys = []
    masks = np.zeros((606, h, w, 1))
    for i, ids in tqdm(enumerate(temp["id"])):
        keys.append(ids)
        annots = train.loc[:, "annotation"][train["id"] == ids].to_list()
        for annot in annots:
            masks[i, :, :] += rle_mask(annot, h, w)
    return masks

def read_images():
    """
    :return: train images of shape (606520,704,1)
    """
    train_images = np.zeros((606, 520,703, 1))
    k = []
    for i, img in tqdm(enumerate(images)):
        k.append(Path(img).stem)
        o = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        o = cv2.resize(o, (256, 256))
        o = np.reshape(o, (256, 256, 1))
        train_images[i, :, :, :] = o
    return train_images

def dataset():
    # read_masks
    masks = read_masks(temp,520,704)
    train_images = read_images()
    return train_images,masks









