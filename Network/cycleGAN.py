from Network.GAN import *

class cycleGAN(GANv2):

    def __build_net__(self):
        self.X = tf.placeholder(shape=[None] + self.input_shape, dtype=tf.float32, name='X')
        self.Y = tf.placeholder(shape=[None] + self.input_shape, dtype=tf.float32, name='Y')
        self.lambdas = tf.placeholder(shape=(), dtype=tf.float32, name='lambda')
        self.dynamic_l_r = tf.placeholder(shape=(), dtype=tf.float32, name='d_l_r')
        self.dropout = tf.placeholder(shape=(), dtype=tf.float32, name='dropout')

        self.G = self.Generator(self.input_shape, self.input_shape, name='G')
        self.F = self.Generator(self.input_shape, self.input_shape, name='F')
        self.Dy = self.Discriminator(self.input_shape, 1, name='Dy')
        self.Dx = self.Discriminator(self.input_shape, 1, name='Dx')

        self.X2Y = self.G(self.X)
        self.Y2X = self.F(self.Y)

        self.X2Y2X = self.F(self.X2Y)
        self.Y2X2Y = self.G(self.Y2X)

        self.realDy = self.Dy(self.Y)
        self.fakeDy = self.Dy(self.X2Y)

        self.realDx = self.Dx(self.X)
        self.fakeDx = self.Dx(self.Y2X)

        self.__set_loss_and_optim__()
        pass

    def __set_loss_and_optim__(self):
        ones_like_Dx = tf.ones_like(self.realDx)
        zeros_like_Dx = tf.zeros_like(self.realDx)

        ones_like_Dy = tf.ones_like(self.realDy)
        zeros_like_Dy = tf.zeros_like(self.realDy)

        LyAdv = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fakeDy, labels=ones_like_Dy))
        LxAdv = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fakeDx, labels=ones_like_Dx))

        LganG = \
            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.realDy, labels=ones_like_Dy)) \
            + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fakeDy, labels=zeros_like_Dy))
        LganF = \
            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.realDx, labels=ones_like_Dx)) \
            + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fakeDx, labels=zeros_like_Dx))

        LcycG = tf.reduce_mean(tf.math.abs(self.X2Y2X - self.X)) * self.lambdas
        LcycF = tf.reduce_mean(tf.math.abs(self.Y2X2Y - self.Y)) * self.lambdas

        LidentityX = tf.reduce_mean(tf.math.abs(self.X - self.X2Y))
        LidentityY = tf.reduce_mean(tf.math.abs(self.Y - self.Y2X))

        self.LDy = LganG# + LcycG
        self.LGy = LyAdv + LcycG + LidentityX

        self.LDx = LganF# + LcycF
        self.LGx = LxAdv + LcycF + LidentityY

        self.TotalDLoss = self.LDy + self.LDx
        self.TotalGLoss = self.LGy + self.LGx

        Dy_vars = self.Dy.trainable_variables
        Dx_vars = self.Dx.trainable_variables

        G_vars = self.G.trainable_variables
        F_vars = self.F.trainable_variables

        self.DyOptim = \
            tf.train.RMSPropOptimizer(learning_rate=self.dynamic_l_r).minimize(self.LDy, var_list=Dy_vars + Dx_vars)
        # self.GyOptim = \
        #     tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.LGy, var_list=G_vars)

        self.DxOptim = \
            tf.train.RMSPropOptimizer(learning_rate=self.dynamic_l_r).minimize(self.LDx, var_list=Dx_vars + Dy_vars)
        # self.GxOptim = \
        #     tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.LGx, var_list=F_vars)
        self.G_optim = \
            tf.train.AdamOptimizer(learning_rate=self.dynamic_l_r)\
                .minimize(self.TotalGLoss, var_list=F_vars + G_vars)

        pass

    def Generator(self,z_shape, output_dim, name= 'generator'):
        #z must be 4D-Tensor
        z = tf.keras.Input(shape=z_shape)
        c7s1_32 = tf_utils.ReflectionPadding2D((3,3))(z)
        c7s1_32 = tf.keras.layers.Conv2D(32, (7,7), (1,1) ,activation=tf.nn.relu)(c7s1_32)

        d64 = tf.keras.layers.Conv2D(64,  (3,3), (2,2), padding='same')(c7s1_32)
        d64 = tf.keras.layers.BatchNormalization()(d64)
        d64 = tf.keras.layers.ReLU()(d64)

        d128 = tf.keras.layers.Conv2D(128, (3,3), (2,2), padding='same')(d64)
        d128 = tf.keras.layers.BatchNormalization()(d128)
        d128 = tf.keras.layers.Dropout(self.dropout)(d128)
        d128 = tf.keras.layers.ReLU()(d128)

        r128 = tf.keras.layers.Conv2D(128, (3,3), (1,1), padding='same')(d128)
        r128 = tf.keras.layers.BatchNormalization()(r128)
        r128 = tf.keras.layers.Dropout(self.dropout)(r128)
        r128 = tf.keras.layers.ReLU()(r128)
        r128 = tf.keras.layers.Conv2D(128, (3,3), (1,1), padding='same')(r128)
        r128 = tf.keras.layers.BatchNormalization()(r128)
        tf.keras.layers.Dropout(self.dropout)(r128)
        r128 = tf.keras.layers.Add()([r128, d128])

        for i in range(8):
            r128_tmp = tf.keras.layers.Conv2D(128, (3,3), (1,1), padding='same')(r128)
            r128_tmp = tf.keras.layers.BatchNormalization()(r128_tmp)
            r128_tmp = tf.keras.layers.Dropout(self.dropout)(r128_tmp)
            r128_tmp = tf.keras.layers.ReLU()(r128_tmp)
            r128_tmp = tf.keras.layers.Conv2D(128, (3,3), (1,1), padding='same')(r128_tmp)
            r128_tmp = tf.keras.layers.BatchNormalization()(r128_tmp)
            r128_tmp = tf.keras.layers.Dropout(self.dropout)(r128_tmp)
            r128 = tf.keras.layers.Add()([r128, r128_tmp])

        u64 = tf.keras.layers.Conv2DTranspose(64, (3,3), (2,2), padding='same')(r128)
        u64 = tf.keras.layers.BatchNormalization()(u64)
        u64 = tf.keras.layers.Dropout(self.dropout)(u64)
        u64 = tf.keras.layers.ReLU()(u64)

        u32 = tf.keras.layers.Conv2DTranspose(32, (3,3), (2,2), padding='same')(u64)
        u32 = tf.keras.layers.BatchNormalization()(u32)
        u32 = tf.keras.layers.Dropout(self.dropout)(u32)
        u32 = tf.keras.layers.ReLU()(u32)

        u32 = tf_utils.ReflectionPadding2D((3,3))(u32)
        c7s1_3 = tf.keras.layers.Conv2D(3, (7,7), (1,1), name=name, activation=tf.nn.tanh)(u32)
        #
        # c7s1_3 = tf.keras.layers.Flatten()(c7s1_3)
        # c7s1_3 = tf.keras.layers.Dense(10, name=name, activation=tf.nn.tanh)(c7s1_3)
        G = tf.keras.Model(z, c7s1_3)
        return G

    def Discriminator(self, input_shape, output_dim, name = 'discriminator'):
        inputs = tf.keras.Input(shape=input_shape)
        c64 = tf.keras.layers.Conv2D(64, (4, 4), (2, 2), activation=tf.nn.leaky_relu, padding='same')(inputs)

        c128 = tf.keras.layers.Conv2D(128, (4, 4), (2, 2), padding='same')(c64)
        c128 = tf.keras.layers.BatchNormalization()(c128)
        c128 = tf.keras.layers.Dropout(self.dropout)(c128)
        c128 = tf.keras.layers.LeakyReLU()(c128)

        c256 = tf.keras.layers.Conv2D(256, (4, 4), (2, 2), padding='same')(c128)
        c256 = tf.keras.layers.BatchNormalization()(c256)
        c256 = tf.keras.layers.Dropout(self.dropout)(c256)
        c256 = tf.keras.layers.LeakyReLU()(c256)

        c512 = tf.keras.layers.Conv2D(512, (4, 4), (2, 2), padding='same')(c256)
        c512 = tf.keras.layers.BatchNormalization()(c512)
        c512 = tf.keras.layers.Dropout(self.dropout)(c512)
        c512 = tf.keras.layers.LeakyReLU()(c512)

        last_layer = tf.keras.layers.Dense(1, activation=None)(c512)
        last_layer = tf.keras.layers.Flatten(name=name)(last_layer)

        #las_conv = tf.keras.layers.Conv2D(1, (4,4), strides=1, padding='same')


        D = tf.keras.Model(inputs, last_layer)
        return D

    def train(self, x= None, z=None, y=None): #z: l_r decay
            x=np.array(x)
            y=np.array(y)
            x=processing.preprocess_image(x)
            y=processing.preprocess_image(y)

            lambdas = 10
            dropout = 0.5

            if z is not None:
                self.learning_rate -= z

            _, Dloss = self.sess.run(
                [self.DyOptim, self.LDy],
                feed_dict={
                    self.X:x,
                    self.lambdas: lambdas,
                    self.Y:y,
                    self.dynamic_l_r: self.learning_rate,
                    self.dropout: dropout
                }
            )
            _, Dloss = self.sess.run(
                [self.DxOptim, self.LDx],
                feed_dict={
                    self.Y: y,
                    self.lambdas: lambdas,
                    self.X: x,
                    self.dynamic_l_r: self.learning_rate,
                    self.dropout: dropout
                }
            )

            for i in range(2):
                _,Gloss = self.sess.run(
                    [self.G_optim, self.TotalGLoss],
                    feed_dict={
                        self.X:x,
                        self.lambdas: lambdas,
                        self.Y:y,
                        self.dynamic_l_r: self.learning_rate,
                        self.dropout: dropout
                    }
                )
            # loss_dict['Y'] = [Dloss,Gloss]
            Dloss, Gloss = self.sess.run(
                [self.TotalDLoss, self.TotalGLoss],
                feed_dict={self.Y: y, self.lambdas: lambdas, self.X:x, self.dropout: dropout}
            )
            return (Dloss,Gloss)

    def generateImg(self, x):
        x = np.array(x)
        x = processing.preprocess_image(x)
        return processing.deprocess_image(self.sess.run(self.X2Y, feed_dict={self.X: x, self.dropout:0})[0])

    def reverseGenerateImg(self, y):
        y =np.array(y)
        y = processing.preprocess_image(y)
        return processing.deprocess_image(self.sess.run(self.Y2X, feed_dict={self.Y: y, self.dropout:0})[0])

    def eval(self, z, y=None):
        pass

    def infer(self, z, y=None, path=None):
        pass


class Mobile(cycleGAN):
    def __build_net__(self):
        self.X = tf.placeholder(shape=[None] + self.input_shape, dtype=tf.float32, name='X')
        self.Y = tf.placeholder(shape=[None] + self.input_shape, dtype=tf.float32, name='Y')
        self.lambdas = tf.placeholder(shape=(), dtype=tf.float32, name='lambda')
        self.dynamic_l_r = tf.placeholder(shape=(), dtype=tf.float32, name='d_l_r')
        self.dropout = 0.0#tf.placeholder(shape=(), dtype=tf.float32, name='dropout')

        self.G = self.Generator(self.input_shape, self.input_shape, name='G')
        self.F = self.Generator(self.input_shape, self.input_shape, name='F')
        self.Dy = self.Discriminator(self.input_shape, 1, name='Dy')
        self.Dx = self.Discriminator(self.input_shape, 1, name='Dx')

        self.X2Y = self.G(self.X)
        self.Y2X = self.F(self.Y)

        self.X2Y2X = self.F(self.X2Y)
        self.Y2X2Y = self.G(self.Y2X)

        self.realDy = self.Dy(self.Y)
        self.fakeDy = self.Dy(self.X2Y)

        self.realDx = self.Dx(self.X)
        self.fakeDx = self.Dx(self.Y2X)

        self.__set_loss_and_optim__()

    def train(self, x= None, z=None, y=None): #z: l_r decay
            x=np.array(x)
            y=np.array(y)
            x=processing.preprocess_image(x)
            y=processing.preprocess_image(y)

            lambdas = 10
            # dropout = 0.5

            if z is not None:
                self.learning_rate -= z

            _, Dloss = self.sess.run(
                [self.DyOptim, self.LDy],
                feed_dict={
                    self.X:x,
                    self.lambdas: lambdas,
                    self.Y:y,
                    self.dynamic_l_r: self.learning_rate,
                    #self.dropout: dropout
                }
            )
            _, Dloss = self.sess.run(
                [self.DxOptim, self.LDx],
                feed_dict={
                    self.Y: y,
                    self.lambdas: lambdas,
                    self.X: x,
                    self.dynamic_l_r: self.learning_rate,
                    #self.dropout: dropout
                }
            )

            for i in range(2):
                _, Gloss = self.sess.run(
                    [self.G_optim, self.TotalGLoss],
                    feed_dict={
                        self.X:x,
                        self.lambdas: lambdas,
                        self.Y:y,
                        self.dynamic_l_r: self.learning_rate,
                        #self.dropout: dropout
                    }
                )
            # loss_dict['Y'] = [Dloss,Gloss]
            Dloss, Gloss = self.sess.run(
                [self.TotalDLoss, self.TotalGLoss],
                feed_dict={self.Y: y, self.lambdas: lambdas, self.X:x}
            )
            return (Dloss,Gloss)

    def Generator(self,z_shape, output_dim, name= 'generator'):
        #z must be 4D-Tensor
        z = tf.keras.Input(shape=z_shape)
        c7s1_32 = tf.keras.layers.ZeroPadding2D((3,3))(z)
        c7s1_32 = tf.keras.layers.Conv2D(32, (7,7), (1,1) ,activation=tf.nn.relu)(c7s1_32)

        d64 = tf.keras.layers.Conv2D(64,  (3,3), (2,2), padding='same')(c7s1_32)
        d64 = tf.keras.layers.BatchNormalization()(d64)
        d64 = tf.keras.layers.ReLU()(d64)

        d128 = tf.keras.layers.Conv2D(128, (3,3), (2,2), padding='same')(d64)
        d128 = tf.keras.layers.BatchNormalization()(d128)
        d128 = tf.keras.layers.Dropout(self.dropout)(d128)
        d128 = tf.keras.layers.ReLU()(d128)

        r128 = tf.keras.layers.Conv2D(128, (3,3), (1,1), padding='same')(d128)
        r128 = tf.keras.layers.BatchNormalization()(r128)
        r128 = tf.keras.layers.Dropout(self.dropout)(r128)
        r128 = tf.keras.layers.ReLU()(r128)
        r128 = tf.keras.layers.Conv2D(128, (3,3), (1,1), padding='same')(r128)
        r128 = tf.keras.layers.BatchNormalization()(r128)
        tf.keras.layers.Dropout(self.dropout)(r128)
        r128 = tf.keras.layers.Add()([r128, d128])

        for i in range(8):
            r128_tmp = tf.keras.layers.Conv2D(128, (3,3), (1,1), padding='same')(r128)
            r128_tmp = tf.keras.layers.BatchNormalization()(r128_tmp)
            r128_tmp = tf.keras.layers.Dropout(self.dropout)(r128_tmp)
            r128_tmp = tf.keras.layers.ReLU()(r128_tmp)
            r128_tmp = tf.keras.layers.Conv2D(128, (3,3), (1,1), padding='same')(r128_tmp)
            r128_tmp = tf.keras.layers.BatchNormalization()(r128_tmp)
            r128_tmp = tf.keras.layers.Dropout(self.dropout)(r128_tmp)
            r128 = tf.keras.layers.Add()([r128, r128_tmp])

        u64 = tf.keras.layers.Conv2DTranspose(64, (3,3), (2,2), padding='same')(r128)
        u64 = tf.keras.layers.BatchNormalization()(u64)
        u64 = tf.keras.layers.Dropout(self.dropout)(u64)
        u64 = tf.keras.layers.ReLU()(u64)

        u32 = tf.keras.layers.Conv2DTranspose(32, (3,3), (2,2), padding='same')(u64)
        u32 = tf.keras.layers.BatchNormalization()(u32)
        u32 = tf.keras.layers.Dropout(self.dropout)(u32)
        u32 = tf.keras.layers.ReLU()(u32)

        u32 = tf.keras.layers.ZeroPadding2D((3,3))(u32)
        c7s1_3 = tf.keras.layers.Conv2D(3, (7,7), (1,1), name=name, activation=tf.nn.tanh)(u32)
        #
        # c7s1_3 = tf.keras.layers.Flatten()(c7s1_3)
        # c7s1_3 = tf.keras.layers.Dense(10, name=name, activation=tf.nn.tanh)(c7s1_3)
        G = tf.keras.Model(z, c7s1_3)
        return G

    def generateImg(self, x):
        x = np.array(x)
        x = processing.preprocess_image(x)
        return processing.deprocess_image(self.sess.run(self.X2Y, feed_dict={self.X: x})[0])

    def reverseGenerateImg(self, y):
        y =np.array(y)
        y = processing.preprocess_image(y)
        return processing.deprocess_image(self.sess.run(self.Y2X, feed_dict={self.Y: y})[0])
