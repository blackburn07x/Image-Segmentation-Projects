import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D,Conv2DTranspose,MaxPool2D
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from data import *
from utils import *
from unet import *

#read_data
train_images,train_masks = dataset()
print(train_images.shape)
print(train_masks.shape)

unet = UNET(1)
ninput = tf.keras.layers.Input(shape = (256,256,1))
logits = unet.UNet(ninput)
model = tf.keras.Model(inputs=  ninput,outputs = logits)

#optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

#define train step
def train_step(x_batch,y_batch):
    with tf.GradientTape() as tape:
        #compute output
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        logits = model(x_batch)
        loss = cross_entropy(logits,y_batch)

    unet_grads = tape.gradient(loss,model.trainable_weights)
    #update
    optimizer.apply_gradients(zip(unet_grads,model.trainable_weights))
    return loss

def train(train_images,train_masks):
    epochs = 100
    batch_size = 64
    num_batches = int(len(train_images)/batch_size)
    for i in range(epochs):
        batch_loss = 0.0
        mini_batches = unet.mini_batches_(train_images,train_masks,batch_size)
        for mini_batch in mini_batches:
            x_batch,y_batch = mini_batch
            loss = train_step(x_batch,y_batch)
            if i%5 ==0:
                print("Train_LOSS: {}".format(loss))
                #get output
                output = model(x_batch,training = False)
                for i in range(4):
                    plt.figure(figsize=(10, 10))
                    # define subplot
                    plt.subplot(2, 2, 1 + i)
                    # turn off axis labels
                    plt.axis('off')
                    # plot single image
                    plt.imshow(output[i], cmap="gray")