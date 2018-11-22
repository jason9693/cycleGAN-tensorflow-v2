import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from Network.cycleGAN import *
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

batch_size = 32
learning_rate = 1e-3
epoch = 100
# label_size = 10
z_dim = 96

sess = tf.Session()
gan = cycleGAN(
        input_shape=[512,512,3],
        learning_rate=learning_rate,
        sess= sess,
        noise_dim=96,
        ckpt_path=None,
        num_classes = 1
)

