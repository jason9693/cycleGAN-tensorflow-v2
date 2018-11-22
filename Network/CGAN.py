import tensorflow as tf
from util import tf_utils, processing
import pprint
import numpy as np
from Network.GAN import *
import re

class CGAN(GAN):   #Conditional GAN Model
    def __init__(self, input_shape, learning_rate, noise_dim, num_class=1,num_labels = 10, sess=None, ckpt_path=None, net='cgan'):
        self.num_labels = num_labels
        super().__init__(input_shape, learning_rate, noise_dim, num_class, sess, ckpt_path, net)

    def __build_net__(self):
        self.X = tf.placeholder(shape=[None]+self.input_shape, dtype=tf.float32, name='X')
        self.Z = tf.placeholder(shape=[None, self.noise_dim], dtype=tf.float32, name='random_z')
        self.y = tf.placeholder(shape=[None], dtype=tf.uint8, name='input_Y')
        self.Y = tf.one_hot(indices=self.y, depth=self.num_labels, dtype=tf.float32, axis=-1)

        self.G = self.Generator(tf.concat([self.Z,self.Y],-1) , self.X.shape[1])

        self.D = self.Discriminator(tf.concat([self.X,self.Y],-1), self.num_classes)
        self.D_G = self.Discriminator(tf.concat([self.G,self.Y],-1), self.num_classes)

        self.__set_loss_and_optim__()

        return

    def train(self, y= None, x= None, z=None):
        if x is None:
            return self.sess.run([self.G_optim, self.G_loss], feed_dict = {
                self.Z: z,
                self.y: y,
                self.X: np.zeros(shape=[y.shape[0], self.input_shape[0]])
            })[1]
        else:
            x = processing.img_preprocessing(x)
            return self.sess.run([self.D_optim, self.D_loss], feed_dict = {
                self.X: x,
                self.Z: z,
                self.y: y,
            })[1]

    def eval(self, z, y=None):
        out = self.sess.run(self.G, feed_dict = {self.Z: z, self.y: y, self.X: np.zeros([self.y[0], self.input_shape[0]])})
        out = processing.img_deprocessing(out)
        return processing.show_images(out)

    def infer(self, z, y=None, path = 'generated/cgan.png'):
        fig = self.eval(z, y)
        return fig.savefig(path)


class infoGAN(DCGAN):
    def __init__(self, input_shape, learning_rate, noise_dim, num_class=1,num_discrete = 10, num_continuous=2,sess=None, ckpt_path=None, net='cgan'):
        self.num_discrete = num_discrete
        self.num_continuous = num_continuous
        super().__init__(input_shape, learning_rate, noise_dim, num_class, sess, ckpt_path, net)

    def __build_net__(self):
        with tf_utils.set_device_mode(par.gpu_mode):
            self.X = tf.placeholder(shape=[None] + self.input_shape, dtype=tf.float32, name='X')
            self.Z = tf.placeholder(shape=[None, self.noise_dim], dtype=tf.float32, name='random_z')
            self.c_dis = tf.placeholder(shape=[None], dtype=tf.uint8, name='input_c_disc')
            self.c_cont = tf.placeholder(shape=[None,self.num_continuous], dtype=tf.float32, name='input_c_cont')
            
        with tf_utils.set_device_mode(False):    
            self.C_DISC = tf.one_hot(indices=self.c_dis, depth=self.num_discrete, dtype=tf.float32, axis=-1)
            #TODO: erase
            print(self.C_DISC.shape)
            
            latent = tf.concat([self.C_DISC, self.Z, self.c_cont], 1)
            
        with tf_utils.set_device_mode(par.gpu_mode):
            latent = latent
            self.G = self.Generator(latent, self.X.shape[1])

            self.D, _, _ = self.Discriminator(self.X, self.num_classes)
            self.D_G,self.Qdis, self.Qcont_params = self.Discriminator(self.G, self.num_classes)
        
        #with tf_utils.set_device_mode(False):
            self.__set_loss_and_optim__()
            return

    def Discriminator(self, input, output_dim, name = 'discriminator'):
        with tf.variable_scope(name,reuse= tf.AUTO_REUSE):# and tf_utils.set_device_mode(par.gpu_mode):
            input = tf.reshape(input, [-1, 28,28,1])

            l0 = tf.layers.conv2d(input, 1, [5,5], strides=(1,1), padding='same')

            l1 = tf.layers.conv2d(l0, self.init_filter_size//4, [5,5], strides=(2,2), padding='same', activation=tf.nn.leaky_relu)
            l1 = tf.layers.batch_normalization(l1)

            l2 = tf.layers.conv2d(
                l1,
                self.init_filter_size//2,
                [5,5],
                strides=(2,2),
                padding='same',
                activation=tf.nn.leaky_relu
            )
            l2 = tf.layers.batch_normalization(l2)

            l3 = tf.layers.flatten(l2)
            l3 = tf_utils.Dense(l3, 64, 'l3', activation=tf.nn.leaky_relu)

            logits = tf_utils.Dense(l3, output_dim, 'logits')
            #print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator'))

        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):# and tf_utils.set_device_mode(par.gpu_mode):
            l3 = tf_utils.Dense(l3, 128, 'q_l3', activation=tf.nn.leaky_relu)
            l3 = tf.layers.batch_normalization(l3, name='q_l3_batch_norm')
            
            logits_dis = tf_utils.Dense(l3, 60, 'q_dis_logits', activation = tf.nn.leaky_relu)
            logits_dis = tf_utils.Dense(logits_dis, self.num_discrete, 'q_dis_logits_final')
            logists_cont = tf_utils.Dense(l3, self.num_continuous * 2, 'q_cont_logits')
            mu = logists_cont[:,:self.num_continuous]
            sigma = logists_cont[:,self.num_continuous:]
            sigma = 1e-8 + tf.nn.softplus(sigma)
            #print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator'))
        return logits, logits_dis, (mu, sigma)

    def __set_loss_and_optim__(self):
        logits_real = tf.ones_like(self.D_G)
        logits_fake = tf.zeros_like(self.D_G)

        self.G_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_G, labels=logits_real)
        self.G_loss = tf.reduce_mean(self.G_loss)

        self.Q_disc_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.Qdis, labels=self.C_DISC)
        self.Q_disc_loss = tf.reduce_mean(self.Q_disc_loss) * 2

        self.Q_cont_loss = +0.5 * tf.log(tf.square(self.Qcont_params[1])+1e-8)\
                           +tf.square(self.c_cont - self.Qcont_params[0]) / (2* tf.square(self.Qcont_params[1]))
        self.Q_cont_loss = tf.reduce_sum(self.Q_cont_loss, 1)
        self.Q_cont_loss = tf.reduce_mean(self.Q_cont_loss) * 0.1

        self.G_loss += self.Q_cont_loss

        self.G_loss += self.Q_disc_loss

        self.D_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D, labels=logits_real) \
                      + tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_G, labels=logits_fake)# + self.Q_cont_loss + self.Q_disc_loss
        self.D_loss = tf.reduce_mean(self.D_loss)

        D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
        print('Discriminator variables: ', D_vars)

        print('\nGenerator variables: ', G_vars)

        self.D_optim = \
            tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.D_loss, var_list=D_vars)
        self.G_optim = \
            tf.train.AdamOptimizer(self.learning_rate).minimize(self.G_loss, var_list=G_vars)
        return

    def train(self, x= None, z=None, y=None):
        if x is None:
            disc, cont = self.sess.run([self.Q_disc_loss, self.Q_cont_loss], feed_dict = {
                self.Z: z[:, :self.noise_dim],
                self.c_dis: y,
                self.c_cont: z[:,self.noise_dim:]
            })
            #print('Q_loss 1. discrete:{} 2. continuous: {}'.format(disc,cont))
            return self.sess.run([self.G_optim, self.G_loss], feed_dict = {
                self.Z: z[:,:self.noise_dim],
                self.c_dis: y,
                self.c_cont: z[:,self.noise_dim:]
            })[1]
        else:
            x = processing.img_preprocessing(x)
            return self.sess.run([self.D_optim, self.D_loss], feed_dict = {
                self.X: x,
                self.Z: z[:, :self.noise_dim],
                self.c_dis: y,
                self.c_cont: z[:,self.noise_dim:]
            })[1]

    def eval(self, z, y=None):
        out = self.sess.run(self.G, feed_dict = {
            self.Z: z[:, :self.noise_dim],
            self.c_dis: y,
            self.c_cont: z[:,self.noise_dim:]
        })
        out = processing.img_deprocessing(out)
        fig = processing.show_images(out,'generated/save.png')

        return fig

    def infer(self, z, y=None, path=None):
        fig = self.eval(z,y)
        fig.savefig('generated/{}.png'.format(self.net))
        return


class SGAN(CGAN):  # Stacked GAN
    def __init__(self,
                 input_shape,
                 learning_rate,
                 noise_dim,
                 lambdas,
                 num_class=1,
                 num_labels = 10,
                 sess=None,
                 ckpt_path=None,
                 num_stack=3,
                 net='SGAN'):
        self.g_lambdas = lambdas
        self.num_stack = num_stack
        super().__init__(input_shape, learning_rate, noise_dim, num_class, num_labels, sess, ckpt_path, net)

    def __build_net__(self):
        self.X = tf.placeholder(shape=[None] + self.input_shape, dtype=tf.float32, name='X')
        self.Z = tf.placeholder(shape=[self.num_stack, None, self.noise_dim], dtype=tf.float32, name='random_z')
        self.y = tf.placeholder(shape=[None], dtype=tf.uint8, name='input_Y')
        self.Y = tf.one_hot(indices=self.y, depth=self.num_labels, dtype=tf.float32, axis=-1)

        x = self.X
        y = self.Y

        self.y_queue = [y]
        self.x_stack = [x]

        for i in range(self.num_stack):
            x, y = self.__build_2_wall__(i, x, y, self.Z[i])
            self.x_stack = [x] + self.x_stack
            self.y_queue.append(y)

        self.__set_loss_and_optim__()
        return

    def __set_loss_and_optim__(self):
        totalLd = 0
        totalLg = 0
        Lg_list = []
        #Lg_list, Ld_list = [], []
        for i in range(self.num_stack):
            Ld, Lg_tuple = \
                self.module_loss(
                    self.x_stack[i+1], self.x_stack[i], self.y_queue[i+1], self.Z[i],
                    name='module'+str(self.num_stack-1-i)
                )
            totalLd = Ld + totalLd
            totalLg = Lg_tuple[0] + Lg_tuple[1] + Lg_tuple[2] + totalLg
            Lg_list.append(Lg_tuple)

        self.D_loss = totalLd
        self.G_loss = totalLg
        self.Lg_seperate = tf.reduce_sum(Lg_list, axis=0)

        D_vars = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if re.search("discriminator", v.name)]
        #tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        G_vars = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if re.search("generator", v.name)]
        #tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
        E_vars = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if re.search("encoder", v.name)]
        #tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'encoder')

        print('all vars:')
        pprint.pprint(tf.global_variables())
        #print([v.name for v in tf.global_variables()])
        print('-----------')
        print('Discriminator variables: ')
        pprint.pprint(D_vars)
        print('-----------')
        print('\nGenerator & Encoder variables: ')
        pprint.pprint(G_vars + E_vars)
        print('===========')

        self.D_optim = \
            tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.D_loss, var_list=D_vars + E_vars)
        self.G_optim = \
            tf.train.AdamOptimizer(self.learning_rate).minimize(self.G_loss, var_list=G_vars + E_vars)
        return

    def module_loss(self, in_e, out_e, out_g, noise, name='module'):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

            out_g = self.Generator(tf.concat([noise,tf.identity(out_e)],1), out_g.shape[1])

            D = self.Discriminator(in_e, self.num_classes)
            D_G = self.Discriminator(out_g, self.num_classes)

            e_hat = self.encoder(out_g, noise.shape[1] * 2, out_e.shape[1] , name='encoder')

            logits_real = tf.ones_like(D_G)
            logits_fake = tf.zeros_like(D_G)

            if name is 'module'+str(self.num_stack-1):
                Lcond = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=e_hat, labels=self.Y))
            else:
                Lcond = tf.reduce_mean(tf.square(tf.nn.softmax(e_hat) - out_e))

            Ladv = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_G,labels=logits_real)
            Ladv = tf.reduce_mean(Ladv)

            recon_z = self.Q(in_e)
            Lent = tf.reduce_mean(tf.square(noise - recon_z))

            Ld = tf.nn.sigmoid_cross_entropy_with_logits(logits=D, labels=logits_real) \
                 + tf.nn.sigmoid_cross_entropy_with_logits(logits=D_G, labels=logits_fake)
            Ld = tf.reduce_mean(Ld) + Lent * 10

        return Ld, (Ladv * self.g_lambdas[0], Lcond * self.g_lambdas[1], Lent * self.g_lambdas[2])

    def encoder(self, h, hidden_dim, output_dim, name='encoder/'):
        l0 = tf_utils.Dense(h, hidden_dim, activation=tf.nn.leaky_relu, name='_l0')
        l1 = tf_utils.Dense(l0, hidden_dim, activation=tf.nn.elu, name=name + '_l1')
        out = tf_utils.Dense(l1, output_dim, activation=None, name=name + 'final')
        return out

    def Q(self, h, disc_name = 'discriminator'):
        h = tf.identity(h)
        #tf.Variable(h, trainable=False, validate_shape=False, collections=)
        D = self.Discriminator(h, self.num_classes, disc_name)
        logits = self.D_L2
        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):  # and tf_utils.set_device_mode(par.gpu_mode):
            l3 = tf_utils.Dense(logits, 128, 'q_l3', activation=tf.nn.leaky_relu)
            l3 = tf.layers.batch_normalization(l3, name='q_l3_batch_norm')

            l4 = tf_utils.Dense(l3, 64, 'q_l4', activation=tf.nn.leaky_relu)
            out = tf_utils.Dense(l4, self.noise_dim, 'q_dis_logits')
            out = tf.nn.tanh(out)
        return out

    def __build_2_wall__(self, cycle, x, y, noise):
        with tf.variable_scope('module' + str(cycle), reuse=tf.AUTO_REUSE):
            if cycle == self.num_stack-1:
                e = tf.nn.softmax(self.encoder(x, noise.shape[1] * 2, self.num_labels, name='encoder'))
            else:
                e = tf.nn.softmax(self.encoder(x, noise.shape[1] * 2, self.input_shape[0], name='encoder'))
        with tf.variable_scope('module' + str(self.num_stack-1 - cycle), reuse=tf.AUTO_REUSE):
            g = self.Generator(tf.concat([noise, y], 1), self.input_shape[0])
        return e, g

    def get_g_losses(self, y, z):
        return self.sess.run(self.Lg_seperate,
                          feed_dict = {self.Z: z, self.y: y, self.X: np.zeros([y.shape[0], self.input_shape[0]])})

    def eval(self, z, y=None):
        out = self.sess.run(
            self.y_queue[self.num_stack],
            feed_dict = {self.Z: z, self.y: y, self.X: np.zeros([len(y), self.input_shape[0]])})
        out = processing.img_deprocessing(out)
        return processing.show_images(out)

    def infer(self, z, y=None, path = 'generated/sgan.png'):
        fig = self.eval(z, y)
        return fig.savefig(path)

#######################################################################
class StackedGAN(CGAN):
    def __init__(self,
                 input_shape,
                 learning_rate,
                 noise_dim,
                 lambdas,
                 num_class=1,
                 num_labels = 10,
                 sess=None,
                 ckpt_path=None,
                 num_stack=3,
                 net='SGAN'):
        self.g_lambdas = lambdas
        self.num_stack = num_stack
        self.init_kernel_size = 7
        self.init_filter_size = 128
        super().__init__(input_shape, learning_rate, noise_dim, num_class, num_labels, sess, ckpt_path, net)

    def __build_net__(self):
        self.X = tf.placeholder(shape=[None] + self.input_shape, dtype=tf.float32, name='X')
        self.Z = tf.placeholder(shape=[self.num_stack, None, self.noise_dim], dtype=tf.float32, name='random_z')
        self.y = tf.placeholder(shape=[None], dtype=tf.uint8, name='input_Y')
        self.Y = tf.one_hot(indices=self.y, depth=self.num_labels, dtype=tf.float32, axis=-1)

        x = self.X
        y = self.Y

        self.y_queue = [y]  #G : G0,G1,...
        self.x_stack = [x]  #E : E2,E1,E0

        self.E = self.encoder(x, self.input_shape[0], self.num_labels)

        for i in range(self.num_stack):
            #y = self.Generator(self.Z[i], self.input_shape[0], final=False)
            x, y = self.__build_wall__(i, x, y, self.Z[i])
            self.x_stack = [x] + self.x_stack
            self.y_queue.append(y)

        print(self.y_queue, self.x_stack)
        self.__set_loss_and_optim__()
        return

    def __set_loss_and_optim__(self):
        totalLd = 0
        totalLg = 0
        Lg_list = []
        #Lg_list, Ld_list = [], []
        for i in range(self.num_stack):
            Ld, Lg_tuple = \
                self.module_loss(
                    self.x_stack[i+1], self.x_stack[i], self.y_queue[i+1], self.Z[i],
                    cycle=self.num_stack-1-i
                )
            totalLd = Ld + totalLd
            totalLg = Lg_tuple[0] + Lg_tuple[1] + Lg_tuple[2] + totalLg
            Lg_list.append(Lg_tuple)

        self.D_loss = totalLd
        self.G_loss = totalLg
        self.Lg_seperate = tf.reduce_sum(Lg_list, axis=0)

        D_vars = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if re.search("discriminator", v.name)]
        G_vars = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if re.search("generator", v.name)]
        E_vars = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if re.search("encoder", v.name)]

        print('all vars:')
        pprint.pprint(tf.global_variables())
        print('-----------')
        print('Discriminator variables: ')
        pprint.pprint(D_vars)
        print('-----------')
        print('\nGenerator & Encoder variables: ')
        pprint.pprint(G_vars + E_vars)
        print('===========')

        self.D_optim = \
            tf.train.RMSPropOptimizer(self.learning_rate * 20).minimize(self.D_loss, var_list=D_vars + E_vars)
        self.G_optim = \
            tf.train.AdamOptimizer(self.learning_rate).minimize(self.G_loss, var_list=G_vars + E_vars)
        return

    def module_loss(self, in_e, out_e, out_g, noise, cycle):
        e_hat = self.encoder(out_g, self.input_shape[0], self.num_labels, recon_layer=cycle)
        #tf_utils.Dense(out_g, out_e.shape[1], name='encoder/l' + str(cycle),
        #    activation=self.encoder_actives[cycle])
        with tf.variable_scope('module'+str(cycle), reuse=tf.AUTO_REUSE):
            out_g = self.Generator(tf.concat([noise,tf.identity(out_e)],1), out_g.shape[1])

            D = self.Discriminator(in_e, self.num_classes)
            D_G = self.Discriminator(out_g, self.num_classes)

            logits_real = tf.ones_like(D_G)
            logits_fake = tf.zeros_like(D_G)

            if cycle == self.num_stack-1:
                Lcond = \
                    tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(
                            logits=e_hat, labels= tf.one_hot(tf.argmax(out_e,1), depth=10)
                        ))
            else:
                Lcond = tf.reduce_mean(tf.square(e_hat - out_e))

            Ladv = tf.nn.sigmoid_cross_entropy_with_logits(logits=D_G,labels=logits_real)
            Ladv = tf.reduce_mean(Ladv)

            recon_z = self.Q(in_e)
            Lent = tf.reduce_mean(tf.square(noise - recon_z))

            Ld = tf.nn.sigmoid_cross_entropy_with_logits(logits=D, labels=logits_real) \
                 + tf.nn.sigmoid_cross_entropy_with_logits(logits=D_G, labels=logits_fake)
            Ld = tf.reduce_mean(Ld) + Lent * 10

        return Ld, (Ladv * self.g_lambdas[0], Lcond * self.g_lambdas[1], Lent * self.g_lambdas[2])

    def encoder(self, h, hidden_dim, output_dim, name='encoder', recon_layer = -1):
        # reconlayer 0: total 1: lay1 2: lay2
        dense_dim = hidden_dim

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            self.encoder_actives = [tf.nn.leaky_relu, tf.nn.elu, None]
            input = h
            if recon_layer <= 1:
                input = tf.reshape(input, [-1, 28, 28, 1])
                input = tf.layers.conv2d(input, 1, [5, 5], strides=(1, 1), padding='same')

                input = tf.layers.conv2d(input, self.init_filter_size // 4, [5, 5], strides=(2, 2), padding='same',
                                     activation=tf.nn.leaky_relu)
                input = tf.layers.batch_normalization(input)
                input = tf.layers.conv2d(
                    input,
                    self.init_filter_size // 2,
                    [5, 5],
                    strides=(2, 2),
                    padding='same',
                    activation=tf.nn.leaky_relu
                )
                input = tf.layers.batch_normalization(input)
                input = tf.layers.flatten(input)
                input = tf_utils.Dense(input, dense_dim, name='reconst', activation=tf.nn.elu)
            for i in range(self.num_stack):
                if recon_layer > 1 and recon_layer-1 == i:
                    input = h

                if i == self.num_stack-1:
                    dense_dim = output_dim
                input = tf_utils.Dense(input, dense_dim, activation=self.encoder_actives[i], name='l'+str(i))

                if i == recon_layer:
                    return input
                #TODO: erase
                print('output shape: {}'.format(input.shape))
        return input

    def Q(self, h, disc_name = 'discriminator'):
        h = tf.identity(h)
        D = self.Discriminator(h, self.num_classes, disc_name)
        logits = self.D_L2
        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):  # and tf_utils.set_device_mode(par.gpu_mode):
            l3 = tf_utils.Dense(logits, 128, 'q_l3', activation=tf.nn.leaky_relu)
            l3 = tf.layers.batch_normalization(l3, name='q_l3_batch_norm')

            l4 = tf_utils.Dense(l3, 64, 'q_l4', activation=tf.nn.elu)
            out = tf_utils.Dense(l4, self.noise_dim, 'q_dis_logits')
            out = tf.nn.tanh(out)
        return out

    def __build_wall__(self, cycle, x, y, noise):
        print('cycle {}: x: {} y: {}'.format(cycle,x.shape,y.shape))
        isFinal = False
        if cycle == self.num_stack - 1:
            isFinal = True

        if isFinal:
            e = tf.nn.softmax(self.encoder(x, self.input_shape[0], self.num_labels, recon_layer=cycle))
            #tf_utils.Dense(x, self.num_labels, name='encoder/l'+str(cycle)))
        else:
            e = self.encoder(x, self.input_shape[0], self.num_labels, recon_layer=cycle)
        with tf.variable_scope('module' + str(self.num_stack-1 - cycle), reuse=tf.AUTO_REUSE):
            g = self.Generator(tf.concat([noise, y], 1), self.input_shape[0], final=isFinal)
        return e, g

    def get_g_losses(self, y, z):
        return self.sess.run(self.Lg_seperate,
                          feed_dict = {self.Z: z, self.y: y, self.X: np.zeros([y.shape[0], self.input_shape[0]])})

    def __fc_layer__(self, input, cycle, name, hidden_dims, sub=None):
        for i in range(cycle):
            activation = tf.nn.leaky_relu
            if type(hidden_dims) == int:
                output_shape = hidden_dims
            else:
                output_shape = hidden_dims[i]
            if i == cycle-1:
                activation = None

            if sub and sub != i:
                continue

            input = tf_utils.Dense(input, output_shape, name=name+'/fc'+str(i), activation=activation)
            if sub:
                return input

        return input


    def Generator(self, z, output_dim, name= 'generator', final = False):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):# and tf_utils.set_device_mode(par.gpu_mode):
            print(z.shape)
            l0 = tf_utils.Dense(
                z,
                self.init_filter_size * self.init_kernel_size * self.init_kernel_size,
                'l0',
                activation=tf.nn.relu
            )
            print(l0.name)
            l0 = tf.reshape(l0, [-1, self.init_kernel_size, self.init_kernel_size, self.init_filter_size])
            l0 = tf.layers.batch_normalization(l0)

            l1 = tf.layers.conv2d_transpose(
                l0,
                self.init_filter_size//2,
                kernel_size=[5,5],
                strides=(2,2),
                padding='same',
                activation=tf.nn.relu
            )
            l1 = tf.layers.batch_normalization(l1)

            if final: activ = tf.nn.tanh
            else: activ = None
            fc = tf.layers.conv2d_transpose(l1, 1, [5,5], strides=(2,2), padding='same', activation=activ)
            fc = tf.layers.flatten(fc)
        return fc

    def Discriminator(self, input, output_dim, name = 'discriminator'):
        with tf.variable_scope(name,reuse= tf.AUTO_REUSE):# and tf_utils.set_device_mode(par.gpu_mode):
            input = tf.reshape(input, [-1, 28,28,1])

            l0 = tf.layers.conv2d(input, 1, [5,5], strides=(1,1), padding='same')

            l1 = tf.layers.conv2d(l0, self.init_filter_size//4, [5,5], strides=(2,2), padding='same', activation=tf.nn.leaky_relu)
            l1 = tf.layers.batch_normalization(l1)

            l2 = tf.layers.conv2d(
                l1,
                self.init_filter_size//2,
                [5,5],
                strides=(2,2),
                padding='same',
                activation=tf.nn.leaky_relu
            )
            l2 = tf.layers.batch_normalization(l2)
            l2 = tf.layers.flatten(l2)
            self.D_L2 = l2
            l3 = tf_utils.Dense(l2, 64, 'l3', activation=tf.nn.leaky_relu)


            logits = tf_utils.Dense(l3, output_dim, 'logits')
            print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator'))
        return logits

    def eval(self, z, y=None):
        out = self.sess.run(
            self.y_queue[self.num_stack],
            feed_dict = {self.Z: z, self.y: y, self.X: np.zeros([len(y), self.input_shape[0]])})
        out = processing.img_deprocessing(out)
        return processing.show_images(out)

    def infer(self, z, y=None, path = 'generated/sgan.png'):
        fig = self.eval(z, y)
        return fig.savefig(path)