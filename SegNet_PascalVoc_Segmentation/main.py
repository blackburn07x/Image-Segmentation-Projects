#libraries
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from network import *
from pascal_voc import *
from utils import *


# train the network
nclasses = 21
epochs = 1000
lr = 1e-4
shape = [128, 128, 3]
path = "vgg16_weights.npz"

# load_dataset
imgs, masks = read_dataset()
trainX, testX, trainY, testY = train_test_split(imgs, masks, test_size= 0.2)

tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, shape[0], shape[1], shape[2]])
Y = tf.placeholder(tf.float32, [None, shape[0], shape[1], nclasses])

# logits and outs
segnet = Segnet(21, path, 64)
pool_out, argmax = segnet.encoder(X)
outputs = segnet.decoder(pool_out, argmax)
logits = outputs

# flat the logits
flat_logits = tf.reshape(logits, [-1, nclasses])
flat_labels = tf.reshape(Y, [-1, nclasses])
outs = tf.nn.softmax(logits)

# sigmoid cross_entropy loss:
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                              labels=flat_labels))

# Optimizers
op = tf.train.AdamOptimizer(lr).minimize(cost)
summary = tf.summary.scalar("Training_Loss__:", cost)

# train the network...
saver = tf.train.Saver()

def mini_batches_(X, Y, batch_size=64):
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
        batch_y = Y[i * batch_size:i * batch_size + batch_size, :, :, :]
        batches.append([batch_x, batch_y])
    return batches


losses = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #saver.restore(sess, "trained_weights")
    # writer = tf.summary.FileWriter(save_summary_dir,graph = tf.get_default_graph())
    for epoch in range(epochs):
        batch_loss = 0.0
        mini_batche = mini_batches_(trainX, trainY, 64)
        for minibatch in mini_batche:
            batch_x, batch_y = minibatch
            loss, summary_b = sess.run([cost, summary], feed_dict={X: batch_x, Y: batch_y})
            _ = sess.run(op, feed_dict={X: batch_x, Y: batch_y})
            batch_loss += loss
            # writer.add_summary(summary_b,epoch)
        average_loss = batch_loss / 64 * 100
        losses.append(average_loss)

        if epoch % 50 == 0 and epoch != 0:
            print("Epoch: {0} ==> \n TRAIN LOSS = {1:0.6f} ".format(epoch + 1, average_loss))
            r = np.random.randint(0, 15)
            mini = mini_batche[r]
            x_test, y_test = mini
            o = sess.run(outs, feed_dict={X: x_test})
            o = np.argmax(o, axis=-1)

            for i in range(9):
                plt.figure(figsize=(10, 10))
                # define subplot
                plt.subplot(3, 3, 1 + i)
                # turn off axis labels
                plt.axis('off')
                # plot single image
                plt.imshow(o[i, :, :], cmap="gray")
            plt.show()

        if epoch % 200 == 0 and epoch != 0:
            print("SAVING MODEL AT {0} epoch.".format(epoch))
            saver.save(sess, "trained/" + str(epoch))
    saver.save(sess, "trained_seg")
    losses = np.array(losses)
    np.save("losses.npy", losses)

pred_images = np.load("outputs/output_500.npy")
orig = np.load("outputs/x_batch_500.npy")
fig,ax = plt.subplots(10,figsize=(20,20))
i=0
for row in range(ax.shape[0]):
    fig.set_tight_layout(True)
    """ax[row, i].set_title("Original Image")
    ax[row, i].imshow(orig[row,:,:])"""
    ax[0].set_title("Predicted Image")
    ax[row].imshow(pred_images[row,:,:])
    i += 1
    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()
fig.savefig('project_fig.png')
plt.imshow(orig[25,:,:])
plt.show()
