import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.model_selection import train_test_split

from utils import *

path = "/content/drive/MyDrive/vgg16_weights.npz"
parameters = np.load(path,encoding='latin1',allow_pickle=True)
weights = {}
keys = sorted(parameters.keys())


class Segnet:

    def __init__(self,classes,vgg_path,batch_size):
        self.classes = classes
        self.vgg_path = vgg_path
        self.batch_size = batch_size

    def encoder(self,x):
        """
        :param x: Input in the network.
        :return: encoder output
        """
        argmax = []
        encoder = ops(self.classes,self.vgg_path)

        #block1
        convblock1 = encoder.conv2dvgg_block(x,1,"SAME","conv1_1_W","conv1_1_b")
        convblock2 = encoder.conv2dvgg_block(convblock1,1,"SAME",'conv1_2_W','conv1_2_b')

        #maxpool1
        pool1,max1,shape1 =encoder.max_pool_(convblock2,"pool1")
        argmax.append(max1)

        #block2
        convblock3 = encoder.conv2dvgg_block(pool1,1,"SAME","conv2_1_W","conv2_1_b")
        convblock4 = encoder.conv2dvgg_block(convblock3,1,"SAME","conv2_2_W",'conv2_2_b')

        #maxpool2
        pool2, max2, shape2 = encoder.max_pool_(convblock4, "pool2")
        argmax.append(max2)

        #block3
        convblock5 = encoder.conv2dvgg_block(pool2,1,"SAME",'conv3_1_W','conv3_1_b')
        convblock6 = encoder.conv2dvgg_block(convblock5, 1, "SAME", 'conv3_2_W', 'conv3_2_b')
        convblock7 = encoder.conv2dvgg_block(convblock6, 1, "SAME", 'conv3_3_W', 'conv3_3_b')

        #maxpool3
        pool3, max3, shape3 = encoder.max_pool_(convblock7, "pool3")
        argmax.append(max3)

        #block4
        convblock8 = encoder.conv2dvgg_block(pool3, 1, "SAME", 'conv4_1_W', 'conv4_1_b')
        convblock9 = encoder.conv2dvgg_block(convblock8, 1, "SAME", 'conv4_2_W', 'conv4_2_b')
        convblock10 = encoder.conv2dvgg_block(convblock9, 1, "SAME", 'conv4_3_W', 'conv4_3_b')

        #maxpool4
        pool4, max4, shape4 = encoder.max_pool_(convblock10, "pool4")
        argmax.append(max4)

        # block5
        convblock11 = encoder.conv2dvgg_block(pool4, 1, "SAME", 'conv5_1_W', 'conv5_1_b')
        convblock12= encoder.conv2dvgg_block(convblock11, 1, "SAME", 'conv5_2_W', 'conv5_2_b')
        convblock13 = encoder.conv2dvgg_block(convblock12, 1, "SAME", 'conv5_3_W', 'conv5_3_b')

        #maxpool5
        pool5, max5, shape5 = encoder.max_pool_(convblock13, "pool4")
        argmax.append(max5)

        return pool5,argmax

    def decoder(self,pool5,argmax):
        """
        :param pool5: output of encoder
        :param argmax: indices of max values
        :return:
        Segnet output
        """

        max1,max2,max3,max4,max5 = argmax

        #decoder
        decoder = ops(self.classes,self.vgg_path)

        #unpool1
        unpool1 = decoder.unpool(pool5,max5,self.batch_size)

        #block1
        deco_conv1 = decoder.conv2d_(unpool1,512,3,1,"SAME","deco_conv1")
        deco_conv2 = decoder.conv2d_(deco_conv1,512,3,1,"SAME","deco_conv2")
        deco_conv3 = decoder.conv2d_(deco_conv2, 512, 3, 1, "SAME","deco_conv3")

        #unpoool2
        unpool2 = decoder.unpool(deco_conv3, max4, self.batch_size)

        deco_conv4 = decoder.conv2d_(unpool2, 512, 3, 1, "SAME", "deco_conv4")
        deco_conv5 = decoder.conv2d_(deco_conv4, 512, 3, 1, "SAME", "deco_conv5")
        deco_conv6 = decoder.conv2d_(deco_conv5, 256, 3, 1, "SAME", "deco_conv6")

        #unpoool3
        unpool3 = decoder.unpool(deco_conv6, max3, self.batch_size)

        deco_conv7 = decoder.conv2d_(unpool3, 256, 3, 1, "SAME", "deco_conv7")
        deco_conv8 = decoder.conv2d_(deco_conv7, 256, 3, 1, "SAME", "deco_conv8")
        deco_conv9 = decoder.conv2d_(deco_conv8, 128, 3, 1, "SAME", "deco_conv9")

        # unpoool4
        unpool4 = decoder.unpool(deco_conv9, max2, self.batch_size)

        deco_conv10 = decoder.conv2d_(unpool4, 128, 3, 1, "SAME", "deco_conv10")
        deco_conv11 = decoder.conv2d_(deco_conv10, 64, 3, 1, "SAME", "deco_conv11")

        # unpoool4
        unpool5 = decoder.unpool(deco_conv11, max1, self.batch_size)

        deco_conv12 = decoder.conv2d_(unpool5, 64, 3, 1, "SAME", "deco_conv12")
        deco_conv13 = decoder.conv2d_(deco_conv12,self.classes,1,1,"VALID","outputs")
        return deco_conv13

ironman