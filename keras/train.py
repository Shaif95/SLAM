

from defination import *

#train and save

ae.fit(x_train, x_train,
                epochs=100,
                batch_size=100,
                shuffle=True,
                validation_split=0.25)

from keras.models import load_model
ae.save('binh.h5')


