import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D,Conv2DTranspose,MaxPool2D
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from utils import *

class SegNet:
    def __init__(self,classes,shape):
        self.classes = classes
        self.HEIGHT = shape[0]
        self.WIDTH = shape[1]
        self.CHANNELS = shape[2]
        self.shape = shape

    def build_network_(self):
        ninput = tf.keras.layers.Input(shape=[256, 256, 1])

        #store the max indices
        argmax=[]
        #encoder Object
        encoder = ops(self.classes)

        #block 1
        convblock1 = encoder.conv2d_block(ninput,3,64,1,"SAME")
        convblock2 = encoder.conv2d_block(convblock1,3,64,1,"SAME")


        pool1,max1 = encoder.maxpoolingwithArgmax(convblock2,2,2,"SAME")
        argmax.append(max1)

        convblock3 = encoder.conv2d_block(pool1,3,128,1,"SAME")
        convblock4 = encoder.conv2d_block(convblock3,3,128,1,"SAME")

        #pool2
        pool2,max2 = encoder.maxpoolingwithArgmax(convblock4,2,2,"SAME")

        argmax.append(max2)

        convblock5 = encoder.conv2d_block(pool2,3,256,1,"SAME")
        convblock6 = encoder.conv2d_block(convblock5,3,256,1,"SAME")
        convblock7 = encoder.conv2d_block(convblock6,3,256,1,"SAME")

        #pool3
        pool3,max3 = encoder.maxpoolingwithArgmax(convblock7,2,2,"SAME")
        argmax.append(max3)

        convblock8 = encoder.conv2d_block(pool3,3,512,1,"SAME")
        convblock9 = encoder.conv2d_block(convblock8,3,512,1,"SAME")
        convblock10 = encoder.conv2d_block(convblock9, 3, 512, 1, "SAME")

        #poool4
        pool4,max4 = encoder.maxpoolingwithArgmax(convblock10,2,2,"SAME")
        argmax.append(max4)

        convblock11 = encoder.conv2d_block(pool4,3,512,1,"SAME")
        convblock12 = encoder.conv2d_block(convblock11,3,512,1,"SAME")
        convblock13 = encoder.conv2d_block(convblock12, 3, 512, 1, "SAME")

        pool5,max5 = encoder.maxpoolingwithArgmax(convblock13,2,2,"SAME")
        argmax.append(max5)

        decoder = ops(self.classes)
        #Un-Pool
        unpool1 = decoder.Unpooling(pool5,max1,2,1)
        print(unpool1.shape)
        #unpool1 = decoder.unpool_with_with_argmax(pool5,max5,2)
        deco_conv1 = decoder.decoder_conv2d_block(unpool1,3,512,1,"SAME")
        deco_conv2 = decoder.decoder_conv2d_block(deco_conv1,3,512,1,"SAME")
        deco_conv3 = decoder.decoder_conv2d_block(deco_conv2,3,512,1,"SAME")

        #unpool2
        unpool2 = decoder.Unpooling(deco_conv3,max2,2,1)
        print(unpool2.shape)
        #unpool2 = decoder.unpool_with_with_argmax(deco_conv3, max4, 2)

        #deco
        deco_conv4 = decoder.decoder_conv2d_block(unpool2,3,512,1,"SAME")
        deco_conv5 = decoder.decoder_conv2d_block(deco_conv4, 3, 512, 1, "SAME")
        deco_conv6 = decoder.decoder_conv2d_block(deco_conv5, 3, 256, 1, "SAME")

        #unpool3
        unpool3 = decoder.Unpooling(deco_conv6,max3,2,1)
        print(unpool3.shape)
        deco_conv7 = decoder.decoder_conv2d_block(unpool3, 3, 256, 1, "SAME")
        deco_conv8 = decoder.decoder_conv2d_block(deco_conv7, 3, 256, 1, "SAME")
        deco_conv9 = decoder.decoder_conv2d_block(deco_conv8, 3, 128, 1, "SAME")

        #unpoool4
        unpool4 = decoder.Unpooling(deco_conv9,max4,2,1)


        #deco
        deco_conv10 = decoder.decoder_conv2d_block(unpool4, 3, 128, 1, "SAME")
        deco_conv11 = decoder.decoder_conv2d_block(deco_conv10, 3, 64, 1, "SAME")

        unpool5 = decoder.Unpooling(deco_conv11,max5,2,1)
        deco_conv12 = decoder.decoder_conv2d_block(unpool5, 3, 64, 1, "SAME")

        #ouput
        deco_conv13 = tf.keras.layers.Conv2D(self.classes,1,1,"VALID")(deco_conv12)
        deco_conv13 = tf.keras.layers.BatchNormalization()(deco_conv13)
        deco_conv13 = tf.keras.layers.Activation("softmax")(deco_conv13)

        seg_model = tf.keras.Model(inputs = ninput,outputs = deco_conv13)
        return seg_model


"""img = cv2.imread("cats.jpg")
img = cv2.resize(img,(224,224))
img = np.reshape(img,(1,224,224,3))
img = np.float32(img)

segnet = SegNet(1,(224,224,3))
inputs = tf.keras.layers.Input(shape = (224,224,3))
out,max = segnet.encoder(inputs)
deco = segnet.decoder(out,max)
model = tf.keras.Model(inputs,deco)
print(model.summary())
"""
from data import *
segnet = SegNet(3,(256,256,1))
model = segnet.build_network_()
model.summary()
train_images,train_masks = dataset()
print(train_images.shape)
print(train_masks.shape)
model(train_images[:10])