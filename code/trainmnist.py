from __future__ import absolute_import, division, print_function, unicode_literals

v = input("Please enter no of epchs:\n")
 
v = int(v)


from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.datasets import mnist
import numpy as np
from tensorflow.keras import layers
from keras import backend as K
from keras.layers import Layer
from keras.layers import Input, Dense



import tensorflow as tf

import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt



import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE

import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from IPython import display


import tensorflow as tf

import os
import cv2
import time
import numpy as np
import glob
import matplotlib.pyplot as plt



import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE

import IPython.display as display
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from IPython import display



from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


from keras.layers import Input, Dense
from keras.models import Model
import keras.layers


def tep1(x):
    
    a= tf.multiply(x,-15.00)
    ex=tf.exp(a)
    b=tf.add(ex,1.00)
    m=tf.truediv(2.00,b)
    n=tf.subtract(1.00,m)
    
    return m;

def tep2(x):
    
    a= tf.multiply(x,-10.00)
    ex=tf.exp(a)
    b=tf.add(ex,1.00)
    m=tf.truediv(2.00,b)
    n=tf.subtract(1.00,m)
    
    return m;

def tep3(x):
    
    a= tf.multiply(x,-2.00)
    ex=tf.exp(a)
    b=tf.add(ex,1.00)
    m=tf.truediv(2.00,b)
    n=tf.subtract(1.00,m)
    return m;


input_img = Input(shape=(784,))
encoded = Dense(256, activation='relu')(input_img)

encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(40, activation=tep3)(encoded)
encoded = Dense(120, activation='relu')(encoded)
encoded = Dense(32, activation=tep1,name='encoder2')(encoded)

decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)
autoencoder = Model(input_img, decoded)
#autoencoder.summary()

from keras import optimizers
adam = optimizers.Adam(lr=0.001)
autoencoder.compile(optimizer=adam, loss='mse')

history=autoencoder.fit(x_train, x_train,
                epochs=v,
                batch_size=256,
                shuffle=True,
                validation_split=.30)
				
				
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder2').output)
encoded_imgs = encoder.predict(x_test)

r=[]
for i in range(len(encoded_imgs)):
	a=np.around(encoded_imgs[i])
	r.append(a)
	
encoded_imgs=np.array(r)

def ham(a):
    s=[]
    n= 32
    for i in range(10000):
        l= np.bitwise_xor(a,z[i])
        t= np.sum(l)
        s.append(t)
    return s;
r=encoded_imgs[20]
s=np.around(r)

z=[]
for q in range (10000):
    s=encoded_imgs[q]
    for l in range(32):
        if(s[l]==2):
            s[l]=s[l]-1
    z.append(s.astype(int))
	
np.savetxt('binary.txt', z, fmt='%d')

from keras.models import load_model


# change the value of e for different digits
e=2   
o=ham(z[e])
b= np.argsort(o)[:10]
data=b
b
plt.imshow(x_test[e].reshape(28,28))
n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(x_test[b[i]].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
