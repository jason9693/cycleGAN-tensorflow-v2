import tensorflow as tf
import tensorflow.contrib as contrib
import numpy as np
import random

def LSTMlayer(input, hidden_size, batch_size,cycle=0):

    lstm_cell = contrib.rnn.LSTMBlockFusedCell(hidden_size,name='lstm'+str(cycle))

    outs,state = lstm_cell(input,dtype=tf.float32)

    return (outs, state)

def Dense(input, output_shape, name='dense' ,activation=None, reuse = tf.AUTO_REUSE):
    with tf.variable_scope(name, reuse=reuse) as scope:
        W = tf.get_variable(
        name='W',
        shape=[input.shape[1], output_shape],
        initializer = contrib.layers.xavier_initializer()
        )
        # #TODO: erase
        # print(W.name)
        # print(W.shape)
        b = tf.get_variable(
        name='b',
        shape=output_shape,
        initializer=contrib.layers.xavier_initializer()
        )

        h = tf.matmul(input, W) + b
        if activation is None:
            return h
        else:
            return activation(h)

def Conv2D(inputs, filters,
    kernel_size,
    strides=(1, 1),
    padding=(0, 0),
    data_format=None,
    dilation_rate=(1, 1),
    activation=None,
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,):
    inputs = tf.pad(
        inputs,
        paddings=((0, 0), (padding[0], padding[0]), (padding[1], padding[1]), (0, 0)),
        mode='CONSTANT')
    conv = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        data_format=data_format,
        dilation_rate=dilation_rate,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        activation=activation)(inputs)

    return conv


def rand_batch(data, batch_size):
    x,y = data
    x = np.array(x,np.float32)
    y = np.array(y,np.int32)
    range_list = np.array(range(x.__len__()))
    batch_index = random.sample(range_list.tolist(), k=batch_size)

    return (x[batch_index], y[batch_index])


def set_device_mode(is_gpu = False):
    gpu_str = 'CPU/:0'
    if is_gpu:
        gpu_str = 'GPU/:0'
    return tf.device(gpu_str)

class ReflectionPadding2D(tf.keras.layers.ZeroPadding2D):
   def call(self, inputs, mask=None):
       #print(self.padding)
       pattern = [[0, 0],
                  self.padding[0],
                  self.padding[1],
                  [0, 0]]
       return tf.pad(inputs, pattern, mode='REFLECT')
