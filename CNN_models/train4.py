import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,Convolution2D,ZeroPadding2D,BatchNormalization
from keras import backend as K
from keras.preprocessing import image
from keras.applications.mobilenet import MobileNet
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.models import Model
import timeit

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
import math
import sys
import time
import datetime
import os



batch_size = 128
num_classes = 7
epochs = 100

class NISTHelper():
    def __init__(self, train_img, train_label, test_img, test_label):
        self.i = 0
        self.test_i = 0
        self.training_images = train_img
        self.training_labels = train_label
        self.test_images = test_img
        self.test_labels = test_label

    def next_batch(self, batch_size):
        x = self.training_images[self.i:self.i + batch_size]
        y = self.training_labels[self.i:self.i + batch_size]
        self.i = (self.i + batch_size) % len(self.training_images)
        return x, y

    def test_batch(self, batch_size):
        x = self.test_images[self.test_i:self.test_i + batch_size]
        y = self.test_labels[self.test_i:self.test_i + batch_size]
        self.test_i = (self.test_i + batch_size) % len(self.test_images)
        return x, y

def unison_shuffled_copies(a, b):
    """Returns 2 unison shuffled copies of array a and b"""
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
def log(logstr):
    """Prints logstr to console with current time"""
    print(datetime.datetime.now().isoformat() + " " + logstr)
    
log("Loading data...")
images = np.load("nist_images_32x32.npy")
labels = np.load("nist_labels_32x32.npy")
log("Data loaded... Shuffling...")
images, labels = unison_shuffled_copies(images, labels)
log("Shuffled!")
split = math.ceil(len(images) * 0.8)
train_imgs = images[:split]
train_labels = labels[:split]
test_imgs = images[split:]
test_labels = labels[split:]
log("Performed train-test split")
print(type(test_imgs))
nist = NISTHelper(train_imgs, train_labels, test_imgs, test_labels)

x = tf.placeholder(tf.float32, shape=[None, 32, 32, 1], name="x")  # Input, shape = ?x32x32x1
y_true = tf.placeholder(tf.float32, shape=[None, 47], name="y_true")  # Label
print()
print(train_imgs.shape)

#Model 4

model = Sequential()
model.add(Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=(32,32,1)))
model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))



model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])

model.fit(train_imgs,train_labels,
          batch_size=batch_size,
          epochs=250,
          verbose=1,
          validation_data=(test_imgs,test_labels))

model.summary()
##incorrects = np.nonzero(model.predict(x).reshape((-1,)) != y_true)
##print(incorrects)
model.save("model.h5")

