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
#from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from IPython import display

import numpy as np
import cv2
import os


#Taking input : Add a 'images' folder with new data . Changing the resize parameters would need changes in other files

images = []


for filepath in os.listdir('images/'):
    images.append(cv2.imread('images/{0}'.format(filepath),0))

print(np.shape(images))

images2 = []


n= len(images)
a = []
b=[]
for i in range (n) :
    a.append(cv2.resize(images[i], dsize=(32,32), interpolation=cv2.INTER_CUBIC))

x_train = np.array(a)

x_train = x_train.astype('float32') / 255.

x_train = np.reshape(x_train, (len(x_train),1024))
print(np.shape(x_train))


x_train = x_train.astype('float32') / 255.

x_train = np.reshape(x_train, (len(x_train),1024))

