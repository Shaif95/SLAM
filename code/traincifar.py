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
from keras.datasets import cifar10
import numpy as np
(x_train, _), (x_test, y_test) = mnist.load_data()

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
np.shape(x_train)
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.




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


from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, concatenate, Conv2DTranspose
from keras.models import Model

inputs = Input((32, 32, 3))
cv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
cv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(cv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(cv1)

cv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
cv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(cv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(cv2)

cv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
cv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(cv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(cv3)

cv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
cv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(cv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(cv4)

cv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
cv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(cv5)


y= Flatten()(cv5)
y= Dense(32, activation=tep3, name='encoder2')(y)

y= Dense(8192, activation='relu')(y)

y = Reshape((4,4,512))(y)


up6 = Conv2D(256, (3, 3), activation='relu', padding='same')(y)
cv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
cv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(cv6)

up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(cv6), cv3], axis=3)
cv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
cv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(cv7)

up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(cv7), cv2], axis=3)
cv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
cv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(cv8)

up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(cv8), cv1], axis=3)
cv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
cv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(cv9)

decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(cv9)

autoencoder = Model(inputs,decoded)
autoencoder.summary()

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


z=[]
for q in range (10000):
    s=encoded_imgs[q]
    for l in range(32):
        if(s[l]==2):
            s[l]=s[l]-1
    z.append(s.astype(int))
	
	

# change the value of e for different digits
e=2   
o=ham(z[e])
b= np.argsort(o)[:10]
data=b
b
plt.imshow(x_test[e].reshape(32, 32,3))
n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(x_test[b[i]].reshape(32, 32,3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

