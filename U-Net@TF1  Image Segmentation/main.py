import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from network import *
from utils import *
from preprocess_data import *


#prepare the data
image_dir = "data/train/"
save_summary_dir = "graph/"
train_images = read_train_images()
train_annots = create_masks()

#save the preprocessed data
np.save("train_images.npy",train_images)
np.save("train_annots.npy",train_annots)

#train_test_split the dataset
trainX,testX,trainY,testY = train_test_split(train_images,train_annots,test_size=0.2)

#train the network
nclasses = 1
epochs = 1
lr = 0.0001
shape = [128, 128, 3]

tf.reset_default_graph()


X = tf.placeholder(tf.float32,[None,shape[0],shape[1],shape[2]])
Y = tf.placeholder(tf.float32,[None,shape[0],shape[1],nclasses])


#logits and outs
unet = UNet(shape,nclasses)
logits = unet.UNet(X)


#flat the logits
flat_logits = tf.reshape(logits,[-1,nclasses])
flat_labels = tf.reshape(Y,[-1,nclasses])
outs = tf.sigmoid(logits)
#sigmoid cross_entropy loss:
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = flat_logits,
                                                              labels = flat_labels))

op = tf.train.AdamOptimizer(lr).minimize(cost)
summary = tf.summary.scalar("Training_Loss__:",cost)

#train the network...
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(save_summary_dir,graph = tf.get_default_graph())
    for epoch in range(epochs):
        batch_loss = 0.0
        mini_batche= unet.mini_batches_(trainX, trainY, 64)
        for minibatch in mini_batche:
            batch_x, batch_y = minibatch
            loss,summary_b = sess.run([cost,summary],feed_dict={X:batch_x,Y:batch_y})
            _ = sess.run(op,feed_dict={X:batch_x,Y:batch_y})
            batch_loss += loss
            writer.add_summary(summary_b,epoch)
        average_loss = batch_loss / 64 * 100

        print("Epoch: {0} ==> \n TRAIN LOSS = {1:0.6f} ".format(epoch + 1, average_loss))
    saver.save(sess,"trained")

#Restore model and make predicition
saver = tf.train.Saver()
with tf.Session() as sess:
  saver.restore(sess,"trained_model/trained")
  #make predictions
  pred = sess.run(outs, feed_dict={X: testX[:100]})

#TEST AND PLOT
def get_true_pred(pred,testY,testX,num_pred = 10):
  predictions =pred[:num_pred,:,:,0]
  true_y  = testY[:num_pred,:,:,0]
  orig_image = testX[:num_pred,:,:]
  return predictions,true_y,orig_image


def show_predictions(preds):
  fig,ax = plt.subplots(10,3,figsize =(20,20))
  predict,true_y,orig = get_true_pred(pred,testY,testX,10)
  i=0
  for row in range(ax.shape[0]):
    fig.set_tight_layout(True)
    ax[row,i].set_title("Original Image")
    ax[row,i].imshow(orig[row]/255.0)
    i+=1
    ax[row,i].set_title("True Output")
    ax[row,i].imshow(true_y[row])
    i+=1
    ax[row,i].set_title("Predicted Image")
    ax[row,i].imshow(predict[row,:,:])
    i=0
    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
  fig.savefig('project_fig.png')

show_predictions(pred)
