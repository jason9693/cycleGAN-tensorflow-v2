from Network.cycleGAN import cycleGAN
from matplotlib import pyplot as plt
import tensorflow as tf
import util.IO as io

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

x_data, x_meta = io.get_datasets('img/X',512)
y_data, y_meta = io.get_datasets('img/Y',512)

loss = gan.train(x=x_data[:5], y=y_data[:5])

print(loss['X'],loss['Y'])

