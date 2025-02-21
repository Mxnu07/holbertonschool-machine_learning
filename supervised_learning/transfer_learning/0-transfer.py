#!/usr/bin/env python3
"""This module trains a CNN to clasiffy CIFRAR10 data set
# using transfer learning using efficient B7"""

from tensorflow import keras as K


def preprocess_data(X, Y):
    X_p = K.applications.efficientnet.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()

# Preprocess the data
x_train, y_train = preprocess_data(x_train, y_train)
x_test, y_test = preprocess_data(x_test, y_test)

# Load the pre-trained EfficientNet-B0 model
base_model = K.applications.EfficientNetB0(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# Add a lambda layer to scale up the image size
model = K.models.Sequential()
model.add(K.layers.Lambda(lambda image: K.backend.resize_images(image, 7, 7, 'channels_last'), input_shape=(32, 32, 3)))
model.add(base_model)

# Unfreeze the last few layers of the pre-trained model
for layer in base_model.layers[:-5]:
    layer.trainable = False

# Add a dense layer for the output
model.add(K.layers.GlobalAveragePooling2D())
model.add(K.layers.Dense(10, activation='softmax'))

# Compile the model
learning_rate_schedule = K.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=10000,
    decay_rate=0.9)
model.compile(optimizer=K.optimizers.Adam(learning_rate=learning_rate_schedule),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Add data augmentation
data_augmentation = K.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)
data_augmentation.fit(x_train)

# Train the model
model.fit(data_augmentation.flow(x_train, y_train, batch_size=64),
          steps_per_epoch=len(x_train) / 64,
          validation_data=(x_test, y_test),
          epochs=30)

# Save the trained model
model.save('cifar10_efficientnetb0.h5')
