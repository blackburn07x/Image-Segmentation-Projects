#libraries
import os
import cv2
import numpy as np
import pandas as pd


HEIGHT = 128
WIDTH = 128
IMG_CHANNELS = 3
MASK_CHANNELS = 1

image_dir = "data/train/"
mask_images_in_order = sorted(os.listdir(image_dir))

rlecodes = pd.read_csv("data/train_masks.csv")

#read rle to mask

def rle2mask(rlecode,SHAPE):
    """
    Function to read rle into images

    :param rlecode: rle-codes of each images from train_masks.csv file
    :param SHAPE: resize shape of image
    :return:
     numpy_array of image masks.
    """
    img = np.zeros(SHAPE[0]*SHAPE[1],dtype = np.float32)
    #start pixels in images
    ffrom = np.array([x for x in rlecode[0:][::2]],dtype=int)
    #rle encoded
    end = np.array([x for x in rlecode[1:][::2]],dtype=int)
    to = ffrom + end
    for f,t in zip(ffrom,to):
        img[f:t]= 1
    img=np.reshape(img,(SHAPE[0],SHAPE[1]))
    return img

#create image masks dataset using rle2mask function
def create_masks():
    """
    Funtion to create mask dataset using the rle2mask function
    :return:
    numpy_array of mask images
    """
    train_annots = np.empty((len(rlecodes),HEIGHT,WIDTH,MASK_CHANNELS),dtype=np.float32)
    for i,rle in enumerate(mask_images_in_order):
        image = cv2.imread(image_dir + rle)
        h,w = image.shape[0],image.shape[1]
        code = rlecodes[rlecodes['img'] == rle]["rle_mask"].iloc[0]
        code = code.split()
        mask_image  = rle2mask(code,(h,w))
        mask_image = cv2.resize(mask_image,(HEIGHT,WIDTH))
        mask_image = np.reshape(mask_image,(HEIGHT,WIDTH,MASK_CHANNELS))
        train_annots[i,:,:,:] = mask_image[:,:]
    return train_annots

#utility function to read images
def read_image(img):
    """
    :param img:
    train images from the img_dir directory
    :return:
    numpy-array of images used for training with HEIGHT,WIDTH
    """

    img = cv2.imread(img)
    img = cv2.resize(img,(HEIGHT,WIDTH))
    return img

#Read train_image
def read_train_images():
    """
    Creates a dataset of images used for training
    :return:
    numpy array dataset of all the images
    """
    train_images = np.empty((len(rlecodes),HEIGHT,WIDTH,IMG_CHANNELS),np.float32)
    for i, image in enumerate(mask_images_in_order):
        img = read_image(image_dir + image)
        train_images[i,:,:,:] = img
    return train_images

