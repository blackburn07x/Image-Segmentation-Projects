import numpy as np
import tensorflow as tf

class ops:
    def __init__(self,classes,vgg_path):
        self.classes = classes
        if vgg_path!=None:
            self.vgg_path = vgg_path

    def imagenet_weights(self, weight_file_path):
        """
        :param weight_file_path: path to weight file
        :return: weights of vgg net
        """
        parameters = np.load(weight_file_path,encoding='latin1',allow_pickle=True)
        self.weights = {}
        keys = sorted(parameters.keys())
        for i in range(len(keys)):
            self.weights[keys[i]] = parameters[keys[i]]
        return self.weights

    def conv2d_vgg(self,x,stride,padding,name_w,name_b):
        """
        :param x: input
              stride: strides in convolutional network
              padding: padding in convolutional network
              name_w: weight_name
              name_b: bias_name
        :return:
        Convolved features

        """
        self.weights = self.imagenet_weights(self.vgg_path)
        with tf.variable_scope(name_w) as scope:
            shape = self.weights[name_w].shape
            initializer = tf.constant_initializer(self.weights[name_w])
            W_ = tf.get_variable(name_w,initializer=initializer,shape = shape)

            initializer_b = tf.constant_initializer(self.weights[name_b])
            B_ = tf.get_variable(name_b,initializer=initializer_b,shape=shape[3])

            conv_ = tf.nn.conv2d(x,W_,strides=[1,stride,stride,1],padding = padding)
            conv_bias = tf.nn.bias_add(conv_,B_)
        return conv_bias

    def conv2dvgg_block(self,x,stride,padding,name_w,name_b):
        """
                :param x: input
                      stride: strides in convolutional network
                      padding: padding in convolutional network
                      name_w: weight_name
                      name_b: bias_name
                :return:
                Convolved features
        """
        conv_bias = self.conv2d_vgg(x,stride,padding,name_w,name_b)
        conv_norm = tf.layers.BatchNormalization()(conv_bias)
        conv_bias_act = tf.nn.relu(conv_norm)
        return  conv_bias_act

    def max_pool_(self,x,name):
        """
        :param x: input
        :return:
        max_pooled input
        """
        with tf.variable_scope(name) as scope:
            pool,argmax = tf.nn.max_pool_with_argmax(x,[1,2,2,1],strides= [1,2,2,1],
                                                     padding="SAME",name = scope.name)
        return pool,argmax,x.get_shape().as_list()

    def unpool(self, pool, ind, batch_size=64):
        """
        :param pool: output of pooling
               ind: max_indices output from tf.nn.max_pool_with_argmax()
               batch_size: size of batch
        :return:
            Unpooled input
        """
        ksize=2
        input_shape = tf.shape(pool)
        output_shape = [input_shape[0], input_shape[1] * ksize, input_shape[2] * ksize, input_shape[3]]
        pool_ = tf.reshape(pool, [-1])
        batch_range = tf.reshape(tf.range(batch_size, dtype=ind.dtype), [tf.shape(pool)[0], 1, 1, 1])
        b = tf.ones_like(ind) * batch_range
        b = tf.reshape(b, [-1, 1])
        ind_ = tf.reshape(ind, [-1, 1])
        ind_ = tf.concat([b, ind_], 1)
        ret = tf.scatter_nd(ind_, pool_, shape=[batch_size, output_shape[1] * output_shape[2] * output_shape[3]])

        ret = tf.reshape(ret, [tf.shape(pool)[0], output_shape[1], output_shape[2], output_shape[3]])
        set_input_shape = pool.get_shape()
        set_output_shape = [set_input_shape[0], set_input_shape[1] * ksize, set_input_shape[2] * ksize,
                            set_input_shape[3]]
        ret.set_shape(set_output_shape)
        return ret

    def conv2d_(self,x,next_filters,filter_size,stride,padding,name):
        """
        :param x:
        :param next_filters: number of output filters
             filter_size: size of each filter hxw
            stride: strides in convolutional network
            padding: padding in convolutional network
            name_w: weight_name
        :return:
        convolved output
        """
        x_shape = x.shape
        filter = [filter_size,filter_size,x_shape[-1],next_filters]
        with tf.variable_scope(name):
            initializer = tf.contrib.layers.xavier_initializer()
            W = tf.get_variable(name,shape = filter,initializer=initializer,dtype = tf.float32)
            B = tf.get_variable(name + "b1",shape = next_filters,initializer=tf.constant_initializer(0.0))
            conv_ = tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding = padding)
            conv_bias_ = tf.nn.bias_add(conv_,B)
            conv_bias_ = tf.layers.BatchNormalization()(conv_bias_)
            conv_bias_act = tf.nn.relu(conv_bias_)
        return conv_bias_act
