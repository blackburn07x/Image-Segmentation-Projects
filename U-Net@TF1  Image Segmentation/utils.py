import tensorflow as tf

#util f()
def downconv_(X,num_filters,filter_size,strides,padding):
    X = tf.layers.conv2d(X,num_filters,filter_size,strides=strides,padding=padding,kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    return X

def upconv_(X,num_filters,filter_size,strides,padding):
    X = tf.layers.conv2d_transpose(X,num_filters,filter_size,strides=strides,padding=padding,kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    return X

def max_pool(X_tensor,ksize,stride=2,padding="SAME"):
    pool = tf.nn.max_pool(X_tensor,[1,ksize,ksize,1],strides=[1,stride,stride,1],padding=padding)
    return pool

