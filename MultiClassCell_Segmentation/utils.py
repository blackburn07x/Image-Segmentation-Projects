import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D

class ops:
    def __init__(self,classes):
        self.classes = classes

    def down_conv_(self,x,out_filters,filter_size,stride,padding):
        conv_ = Conv2D(out_filters,filter_size,strides = (stride,stride),
                           padding = padding, kernel_initializer=tf.initializers.GlorotNormal()
                           )(x)
        batch_norm_ = tf.keras.layers.BatchNormalization()(conv_)
        conv_batch_norm_act = tf.keras.layers.Activation("relu")(batch_norm_)
        return conv_batch_norm_act

    def up_conv_(self,x,out_filters,filter_size,stride,padding):
        up_conv_ = tf.keras.layers.Conv2DTranspose(out_filters,filter_size,strides =(stride,stride),padding = padding)(x)
        batch_norm_ = tf.keras.layers.BatchNormalization()(up_conv_)
        conv_batch_norm_act = tf.keras.layers.Activation("relu")(batch_norm_)
        return conv_batch_norm_act

    def max_pool_(self,x,filter_size,stride=2,padding="SAME"):
        max_pool = tf.keras.layers.MaxPool2D(filter_size,strides = stride,padding=padding)(x)
        return max_pool

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



