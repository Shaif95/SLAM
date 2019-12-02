from data import *
from keras.layers import Input, Dense, Lambda, Activation, Convolution2D, MaxPooling2D
from keras.models import Model, Sequential
from keras import backend as K
from keras.datasets import mnist
from keras.layers.core import Reshape
from keras.layers.convolutional import UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.datasets import mnist
import numpy as np
from tensorflow.keras import layers
from keras import backend as K
from keras.layers import Layer
from keras.layers import Input, Dense
from keras.optimizers import SGD

import tensorflow as tf

import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt

shape=32
batch_size = 30
nb_classes = 10
img_rows, img_cols = shape, shape
nb_filters = 32
pool_size = (2, 2)
kernel_size = (3, 3)
input_shape=(shape,shape,1)
original_dim = 1024
latent_dim = 2
intermediate_dim = 256
epsilon_std = 1.0
learning_rate = 0.028
decay_rate = 5e-5
momentum = 0.9
sgd = SGD(lr=learning_rate,momentum=momentum, decay=decay_rate, nesterov=False)
part=8
thre=1


import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE

import IPython.display as display
#from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from IPython import display

import numpy as np
import cv2
import os
#def

# uses inbuilt regularizers to create sparse code
# might need changes according to the size of data.
# more layers can be added for larger size

from keras import regularizers

encoding_dim = 32

input_img = Input(shape=(1024,))
# add a Dense layer with a L1 activity regularizer

x = Dense(64, activation='tanh')(input_img)

n = Dense(32, activation='tanh',activity_regularizer=regularizers.l1(6e-6), name='encoder' )(x)


h = Dense(64, activation='tanh')(n)

decoded = Dense(1024, activation='sigmoid')(h)

ae = Model(input_img, decoded)
ae.summary()

from keras import optimizers
adam = optimizers.Adam(lr=0.001)
ae = Model(input_img, decoded)
ae.compile(optimizer=adam, loss='mse')


# A discriminator and GAN can be used here to decrease reconstruction loss. 




recog=Sequential()
recog.add(Dense(64,activation='relu',input_shape=(1024,),init='glorot_uniform'))
get_0_layer_output=K.function([recog.layers[0].input, 
                                 K.learning_phase()],[recog.layers[0].output])
c=get_0_layer_output([x_train[0].reshape((1,1024)), 0])[0][0]

recog_left=recog
recog_left.add(Lambda(lambda x: x + np.mean(c), output_shape=(64,)))

recog_right=recog
recog_right.add(Lambda(lambda x: x + K.exp(x / 2) * K.random_normal(shape=(1, 64), mean=0., stddev=epsilon_std), output_shape=(64,)))

recog1=Sequential()


recog1.add(Dense(64, activation='relu',init='glorot_uniform'))
recog1.add(Dense(1024, activation='relu',init='glorot_uniform'))
recog1.compile(loss='mean_squared_error', optimizer=sgd,metrics = ['mae'])



de=Sequential()
de.add(Reshape((32,32,1),input_shape=(1024,)))
de.add(Convolution2D(20, 3,3,
                        border_mode='valid',
                        input_shape=input_shape))
de.add(BatchNormalization())
de.add(Activation('relu'))
de.add(UpSampling2D(size=(2, 2)))
de.add(Convolution2D(20, 3, 3,
                            init='glorot_uniform'))
de.add(BatchNormalization())
de.add(Activation('relu'))
de.add(Convolution2D(20, 3, 3,init='glorot_uniform'))
de.add(BatchNormalization())
de.add(Activation('relu'))
de.add(MaxPooling2D(pool_size=(3,3)))
de.add(Convolution2D(4, 3, 3,init='glorot_uniform'))
de.add(BatchNormalization())
de.add(Activation('relu'))
de.add(Reshape((32,32,1)))
de.add(Reshape((1024,)))
de.add(Dense(1024, activation='sigmoid',init='glorot_uniform'))


def not_train(net, val):
    net.trainable = val
    for k in net.layers:
       k.trainable = val
not_train(recog1, False)

gan_input = Input(batch_shape=(1,1024))

gan_level2 = de(recog1(gan_input))

NN = Model(gan_input, gan_level2)





print("runs")