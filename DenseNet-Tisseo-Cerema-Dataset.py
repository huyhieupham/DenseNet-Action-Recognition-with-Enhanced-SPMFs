# Huy-Hieu PHAM, Ph.D. student
# DenseNet for image recognition.
# Python 3.5.2 using Keras with the Tensorflow Backend.
# Created on 25.01.2018


from __future__ import print_function

import os
import time
import timeit
import json
import argparse
import densenet
import numpy as np
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools
import math

from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, merge
from keras.engine import Input, Model
from keras.optimizers import SGD
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix

# Load and process data.
nb_classes = 8

# learning rate schedule
def step_decay(epoch):
	initial_lrate = 0.001
	drop = 0.1
	epochs_drop = 100.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

# 2539 for training and 2536 for testing.
img_width, img_height = 32, 32
train_data_dir = 'data/tisseo/train' 
validation_data_dir = 'data/tisseo/validation'
nb_train_samples = 2539
nb_validation_samples = 2536
epochs = 250
batch_size = 256

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


    # Construct DenseNet architeture.
    model = densenet.DenseNet(nb_classes,       # Number of classes: 8 for MSR Action3D and 60 for NTU-RGB+D.
                              input_shape,   	# Input_shape.
                              40,				# Depth: int -- how many layers; "Depth must be 3*N + 4"
                              3,				# nb_dense_block: int -- number of dense blocks to add to end
                              12,				# growth_rate: int -- number of filters to add
                              16,				# nb_filter: int -- number of filters
                              dropout_rate=0.2,
                              weight_decay=0.0001)						 
							  
							  
# Model output.
model.summary()

# Compile the model.
model.compile(optimizer=Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])
				  
# learning schedule callback
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]

# Data augmentation.
train_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size = (img_width, img_height),
                                                    batch_size = batch_size,
                                                    class_mode = 'sparse')

validation_generator = test_datagen.flow_from_directory(validation_data_dir,
                                                        target_size = (img_width, img_height),
                                                        batch_size = batch_size,
                                                        class_mode = 'sparse')


# Fit model.
history = model.fit_generator(train_generator,
                              steps_per_epoch=nb_train_samples // batch_size,
                              epochs=epochs,
                              validation_data=validation_generator,
                              validation_steps=nb_validation_samples // batch_size,
                              callbacks=callbacks_list,
							  verbose=2
                              )

# Saving weight.
model.save_weights('DenseNet-40-Tisseo.h5')

# List all data in history.
print(history.history.keys())

# grab the history object dictionary
H = history.history

last_test_acc = history.history['val_acc'][-1] * 100
last_train_loss = history.history['loss'][-1] 
last_test_acc = round(last_test_acc, 2)
last_train_loss = round(last_train_loss, 6)
train_loss = 'Training Loss, min = ' +  str(last_train_loss)
test_acc = 'Test Accuracy, max = ' + str(last_test_acc) +' (%)'
 
# plot the training loss and accuracy
N = np.arange(0, len(H["loss"]))
plt.style.use("ggplot")
plt.figure()
axes = plt.gca()
axes.set_ylim([0.0,1.2])
plt.plot(N, H['loss'],linewidth=2.5,label=train_loss,color='blue')
plt.plot(N, H['val_acc'],linewidth=2.5, label=test_acc,color='red')
#plt.plot(N, H['val_loss'],linewidth=2.5,label="Test Loss")
#plt.plot(N, H['acc'],linewidth=2.5, label="Training Accuracy")
plt.title('Enhanced-SPMF DenseNet-40 on Tisséo',fontsize=12, fontweight='bold',color = 'Gray')
plt.xlabel('Number of Epochs',fontsize=11, fontweight='bold',color = 'Gray')
plt.ylabel('Training Loss and Test Accuracy',fontsize=12, fontweight='bold',color = 'Gray')
plt.legend()
 
# Save the figureL
plt.savefig('output/tisseo/Enhanced-SPMF-DenseNet-40.png')
plt.show()