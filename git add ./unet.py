import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D,Conv2DTranspose,MaxPool2D
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Concatenate

from utils import *

class UNET:
    def __init__(self, classes):
        """
        :param classes: No.of classes
        """
        self.classes = classes
        self.ops = ops(self.classes)

    def down_conv_block(self, x, filters):
        """
        :param x:
        :param filters:

        :return:
        """
        s1 = self.ops.down_conv_(x, filters, filter_size= 3, stride= 1, padding= "SAME")
        s2 = self.ops.down_conv_(s1, filters, filter_size= 3, stride= 1, padding= "SAME")
        return s2

    def up_conv_block(self, x, filters, skip_connection):
        """
        :param x: input
        :param filters: no. of filters
        :param skip_connection:

        """
        e1 = self.ops.up_conv_(x, filters, filter_size= 2, stride= 2, padding= "SAME")
        concat = tf.concat([e1, skip_connection], axis= -1)
        #layer 2
        conv1 = self.ops.down_conv_(concat,filters,filter_size = 3,stride = 1, padding= "SAME")
        #layer3
        conv2 = self.ops.down_conv_(conv1,filters,filter_size = 3,stride = 1, padding = "SAME")
        return conv2

    def UNet(self, x):
        """
        :param x: input

        :return:
        Output of the U-Net
        """
        #encoder
        d1 = self.down_conv_block(x,32)
        m1 = self.ops.max_pool_(d1,filter_size=2,stride= 2,padding= "SAME")
        d2 = self.down_conv_block(m1,64)
        m2 = self.ops.max_pool_(d2,filter_size =2,stride=2,padding="SAME")
        d3 = self.down_conv_block(m2,128)
        m3 = self.ops.max_pool_(d3,filter_size=2,stride= 2,padding = "SAME")
        d4  =self.down_conv_block(m3,256)
        m4 = self.ops.max_pool_(d4,filter_size=2,stride= 2,padding = "SAME")

        #bottleneck
        bridge = self.ops.down_conv_(m4,1024,3,1,"SAME")
        bridge = self.ops.down_conv_(bridge,1024,3,1,"SAME")

        #decoder
        u1 = self.up_conv_block(bridge,256,d4)
        u2 = self.up_conv_block(u1,128,d3)
        u3 = self.up_conv_block(u2,64,d2)
        u4 = self.up_conv_block(u3,32,d1)

        #1x1 output
        logits = tf.keras.layers.Conv2D(self.classes,kernel_size=1,strides=1,padding="SAME")(u4)
        logits = tf.nn.sigmoid(logits)
        return logits

    def mini_batches_(self, X, Y, batch_size=64):
        """
        function to produce minibatches for training
        :param X: input placeholder
        :param Y: mask placeholder
        :param batch_size: size of each batch
        :return:
        minibatches for training
        """
        train_length = len(X)
        num_batches = int(np.floor(train_length / batch_size))
        batches = []
        for i in range(num_batches):
            batch_x = X[i * batch_size: i * batch_size + batch_size, :, :, :]
            batch_y = Y[i * batch_size:i * batch_size + batch_size, :, :]
            batches.append([batch_x, batch_y])
        return batches