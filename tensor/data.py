import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
%matplotlib inline

np.random.seed(0)
tf.set_random_seed(0)

import numpy as np
import cv2
import os

images = []

for filepath in os.listdir('places/'):
    images.append(cv2.imread('places/{0}'.format(filepath),0))
    
n=len(images)

BLUE = [255,255,255]
a = []
for i in range (n):
    j=(cv2.copyMakeBorder(images[i], 32, 32, 16,0, cv2.BORDER_REPLICATE, value=BLUE))
    a.append(cv2.resize(j, dsize=(28,28), interpolation=cv2.INTER_CUBIC))
    
x_train = np.array(a)

x_train = x_train.astype('float32') / 255.

x_train = x_train.reshape((len(x_train), 784))



n_samples = len(x_train)
np.shape(x_train)