import tensorflow as tf
import numpy as np
import cv2

from utils import *

#Define U-net
class UNet:
    def __init__(self,shape,classes):
        #classes
        self.classes=classes

    def dconv_block(self, X_tensor, filters):
        """
        Function to downsample (emcoder) input images.
        :param X_tensor: placeholder for inputs
        :param filters: number of filters to be used
        :return:
        downsampled image
        """
        #layer1
        s1 = downconv_(X_tensor,filters,filter_size= 3,strides=1,padding='SAME')
        b1 = tf.layers.batch_normalization(s1)
        a1 = tf.nn.relu(b1)

        #layer2
        s2 = downconv_(a1,filters,filter_size=3,strides=1,padding='SAME')
        b2 = tf.layers.batch_normalization(s2)
        a2 = tf.nn.relu(b2)
        return a2

    def upconv_block(self,X_tensor,filters,filter_size,skip_connection):
        """
        Function to upsample (transposed-convolution) image and
        use for building decoder part of the network
        :param X_tensor: placeholder for inputs
        :param filters: number of filters to be used
        :param filter_size: size of the filter(kernel) to be used
        :param skip_connection: part of decoder network to stich

        :return:
        upsampled and stiched image
        """
        #layer1
        e1 = upconv_(X_tensor,filters,filter_size=filter_size,strides =2,padding="SAME")
        concat = tf.concat([e1,skip_connection],axis=-1)

        #layer2
        conv1 = downconv_(concat,filters,filter_size=3,strides=1,padding='SAME')
        relu1 = tf.nn.relu(conv1)
        #layer3
        conv2 = downconv_(relu1,filters,filter_size=3,strides=1,padding='SAME')
        relu2 = tf.nn.relu(conv2)

        return relu2

    def UNet(self,X_tensor):
        """
        Encoder-Decoder components of UNet. Loss funtion used is Binary-crossentropy
        filters: [32,64,128,256]
        :param:
        X_tesnor : placeholder for train images (X)

        :return:
        probability masks for each class in the image
        """
        #encoder
        d1 = self.dconv_block(X_tensor, 32)
        m1 = max_pool(d1, ksize=2, stride=2, padding="SAME")
        d2 = self.dconv_block(m1, 64)
        m2 = max_pool(d2, ksize=2, stride=2, padding="SAME")
        d3 = self.dconv_block(m2, 128)
        m3 = max_pool(d3, ksize=2, stride=2, padding="SAME")
        d4 = self.dconv_block(m3,256)
        m4 = max_pool(d4, ksize=2, stride=2, padding="SAME")

        #bottleneck
        bridge = downconv_(m4, 1024, 3, 1, 'SAME')
        bridge = downconv_(bridge, 1024, 3, 1, 'SAME')

        #decoder
        u1 = self.upconv_block(bridge, 256, 2, d4)
        u2 = self.upconv_block(u1, 128, 2, d3)
        u3 = self.upconv_block(u2, 64, 2, d2)
        u4 = self.upconv_block(u3, 32, 2, d1)

        #1x1 output conv
        logits = downconv_(u4,1,self.classes,strides=1,padding="SAME")
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
