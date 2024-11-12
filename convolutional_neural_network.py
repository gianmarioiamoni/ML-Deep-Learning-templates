# CONVOLUTIONAL NEURAL NETWORK

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.__version__

# Part 1 - Data Preprocessing

# Preprocessing the Training set
#
# We will apply transformations to traning set images to avoid overfitting
# This is called Image Augmentation
#
# We use the ImageDataGeneration class
train_datagen = ImageDataGenerator(
    rescale = 1./255, # features scaling
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True)

# connect the train_datagen to the training_set
training_set = train_datagen.flow_from_directory('dataset/training_set',
                  target_size = (64, 64), # final size of the images in the CNN
                  batch_size = 32, # size of the batches (number of images in the batch)
                  class_mode = 'binary')

# Preprocessing the Test set
#
# We maintain the test set intact, without transformation
# but we need to apply features scaling
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'binary')

# Part 2 - Building the CNN

# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 - Convolution
#
# We use the Conv2D class from the layers module of keras library
# unit = n. of filters (based on experimentation, no rule of thumb)
# kernel_size = size of the filter matrix (3x3)
# activation = activation function
# input_shape = shape of the input image (64x64x3)
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

# Step 2 - Pooling
#
# We use the MaxPooling2D class from the layers module of keras library
# pool_size = size of the pooling window (2x2)
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))

# Adding a second convolutional layer
#
# input_shape parmeter is need only in the 1st layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
# Adding a second pooling layer
cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))

# Step 3 - Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full connection
#
# We use the Dense class from the layers module of keras library
# unit = n. of neurons (based on experimentation, no rule of thumb)
# activation = activation function
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output layer
#
# We use the Dense class from the layers module belonging to the keras library of TensorFlow
# units: number of output neurons (binary classification in this case)
# activation: activation function ('sigmoid' for the output layer)
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the CNN

# Compiling the CNN
#
# We use the compile() method of the cnn
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
#
# We use the fit() method of the cnn
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)

# Part 4 - Making a single prediction
#
# We deploy the CNN for each of the 2 prediction images

import numpy as np
from tensorflow.keras.preprocessing import image
# load a single image to which we want to deploy the CNN
# The image must have the same size of the images we used for the training (64x64 in this case)
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
# convert the image in an numpy array, expected by the CNN as input
test_image = image.img_to_array(test_image)
# CNN was trained on batches of images. The single image has to be in a batch
# we add an extra dimension corresponding to the bacth;
# axis: where we want to add the extra dimension (first dimension = 0)
test_image = np.expand_dims(test_image, axis = 0)
# deploy the CNN
result = cnn.predict(test_image)
# print the right class indeces
training_set.class_indices
# encoding the result (result is in a batch, so is a 2d array)
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
    
print(prediction)