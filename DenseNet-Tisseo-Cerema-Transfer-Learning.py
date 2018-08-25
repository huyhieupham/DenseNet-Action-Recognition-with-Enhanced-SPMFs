# Huy-Hieu PHAM, Ph.D student
# Cerema & Institut de Recherche en Informatique de Toulouse (IRIT)
# Description: Training Inception V-4 model from human action recognition.
# Date: 15 / 08 / 2018
# Python 3.5.2, Keras 2.0.8 with Tensorflow backend.



# Import libraries and packages.
import os
import math

import numpy as np
import tensorflow as tf
from keras import optimizers
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, merge
from keras.engine import Input, Model
from keras.layers.advanced_activations import ELU
from keras.optimizers import SGD
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.models import load_model
import keras.backend as K
import json
import time

# Number of action classes on the Tisseo_Cerema dataset.
nb_classes = 3
	
# Learning rate schedule.
def step_decay(epoch):
	initial_lrate = 0.001
	drop = 0.5
	epochs_drop = 20
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


# Load pre-trained model.
model = load_model('DenseNet-40.h5')
# Print the model architeture.
model.summary()

# Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
for layer in model.layers[:5]:
    layer.trainable = False

# Remove some last layers.

new_model_1 = Model(model.inputs, model.layers[-2].output)

new_model_1.summary()

x = new_model_1.output
predictions = Dense(nb_classes, activation="softmax")(x)
new_model_2 = Model(model.inputs, output=predictions)

new_model_2.summary()

# Add more dense layer at the end of the network.




# Compile the model.

new_model_2.compile(optimizer = 'adam',
             loss = 'sparse_categorical_crossentropy',
             metrics = ['accuracy'])


lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]

# Data augmentation.

train_datagen = ImageDataGenerator(
    
                #rotation_range = 40,          # rotation_range is a value in degree from 0 to 180.
                width_shift_range = 0.2,      # shift 20% of total width.
                height_shift_range = 0.2,     # shift 20% of total height.
                rescale = 1./255,              # RGB values are in the range of 0 - 1.
                #shear_range = 0.2,            # shear an image with 20% .
                #zoom_range = 0.2,             # randomly zooming.
                #horizontal_flip = True,       # apply horizontal flipping.
                #fill_mode = 'nearest'         # the strategy used for filling in newly created pixels,
                                               # which can appear after a rotation or a width/height shift.
                )


test_datagen = ImageDataGenerator(rescale = 1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = 'binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = 'binary')


# Fit model.

history = new_model_2.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=callbacks_list,
        verbose=2)
		
# Store trained model for later use
# Delete the model
del model

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
plt.title('Enhanced-SPMF DenseNet-40 on Tisséo with Transfer Learning',fontsize=10, fontweight='bold',color = 'Gray')
plt.xlabel('Number of Epochs',fontsize=10, fontweight='bold',color = 'Gray')
plt.ylabel('Training Loss and Test Accuracy',fontsize=10, fontweight='bold',color = 'Gray')
plt.legend()
 
# Save the figureL
plt.savefig('output/tisseo/DenseNet-40-Transfer-Learning.png')
plt.show()
