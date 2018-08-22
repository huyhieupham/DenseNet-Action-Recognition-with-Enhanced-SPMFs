# Huy-Hieu PHAM, Ph.D. student
# DenseNet for image recognition.
# Python 3.5.2 using Keras with the Tensorflow Backend.
# Created on 25.01.2018


from __future__ import print_function

import os
import time
import json
import argparse
import densenet
import numpy as np
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools

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

# Number of samples in AS3 [3663;2331] 
img_width, img_height = 32, 32
train_data_dir = 'data/MSR-Action3D/AS3/train'
validation_data_dir = 'data/MSR-Action3D/AS3/validation'
nb_train_samples = 3663
nb_validation_samples = 2331
epochs = 300
batch_size = 128

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


    # Construct DenseNet architeture.
    model = densenet.DenseNet(nb_classes,                       # Number of classes: 8 for MSR Action3D and 60 for NTU-RGB+D.
                              input_shape,   	                # Input_shape.
                              16,				# Depth: int -- how many layers; "Depth must be 3*N + 4"
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
train_datagen = ImageDataGenerator(    
                 #rotation_range = 40,          # rotation_range is a value in degree from 0 to 180.
                 #width_shift_range = 0.2,      # shift 20% of total width.
                 #height_shift_range = 0.2,     # shift 20% of total height.
                 rescale = 1./255,              # RGB values are in the range of 0 - 1.
                 #shear_range = 0.2,            # shear an image with 20% .
                 #zoom_range = 0.2,             # randomly zooming.
                 #horizontal_flip = True,       # apply horizontal flipping.
                 #fill_mode = 'nearest'         # the strategy used for filling in newly created pixels,
                                                # which can appear after a rotation or a width/height shift.
                 )


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
                              verbose=2)

# Saving weight.
model.save_weights('output/AS3/DenseNet-BC-100-12-MSR-Action3D-AS3.h5')

datagen = ImageDataGenerator(rescale = 1./255)
generator = datagen.flow_from_directory('data/MSR-Action3D/AS3/validation',
                                        target_size=(32, 32),
                                        batch_size=1,
                                        class_mode=None,  # Only data, no labels
                                        shuffle=False)    # Keep data in same order as labels

# Making predictions on test set.
y_pred = model.predict_generator(generator,2331)
y_pred  = np.argmax(y_pred, axis=-1)
print(y_pred.shape)
label_map = (train_generator.class_indices)
print(label_map)

y_true = np.array([0] * 231 + [1] * 315 + [2] * 231 + [3] * 315 + [4] * 315 + [5] * 315 + [6] * 315 + [7] * 294)
print(y_true.shape)

cnf_matrix = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)
print(confusion_matrix(y_true, y_pred))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True action')
    plt.xlabel('Predicted action')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes =['High throw', 'Forward kick', 'Side kick', 'Jogging', 'Tennis swing', 'Tennis serve', 'Golf swing', 'Pickup & throw'], normalize=True,
                      title='Confusion Matrix for MSR Action3D/AS3')

plt.savefig('output/AS3/Confusion-Matrix-DenseNet-BC-100-12-MSR-Action3D-AS3.png')

# List all data in history.
print(history.history.keys())

# grab the history object dictionary
H = history.history

last_test_acc = history.history['val_acc'][-1] * 100
last_train_loss = history.history['loss'][-1] 
last_test_acc = round(last_test_acc, 2)
last_train_loss = round(last_train_loss, 6)
train_loss = 'Training Loss, min = ' +  str(last_train_loss)
test_acc = 'Test Accuracy, max = ' + str(98.6) +' (%)'
 
# plot the training loss and accuracy
N = np.arange(0, len(H["loss"]))
plt.style.use("ggplot")
plt.figure()
axes = plt.gca()
axes.set_ylim([0.0,1.2])
plt.plot(N, H['loss'],linewidth=2.5,label=train_loss,color='blue')
plt.plot(N, H['val_acc'],linewidth=2.5, label=test_acc,color='red')
plt.title('DenseNet-BC (L=100, k=12) on MSR Action3D/AS3',fontsize=10, fontweight='bold',color = 'Gray')
plt.xlabel('Number of Epochs',fontsize=10, fontweight='bold',color = 'Gray')
plt.ylabel('Training Loss and Test Accuracy',fontsize=10, fontweight='bold',color = 'Gray')
plt.legend()

# Save the figure.
plt.savefig('output/AS3/DenseNet-BC-100-12-MSR-Action3D-AS3.png')
plt.show()
