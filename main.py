import tensorflow as tf 
from tensorflow import keras 
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras import layers, models
import matplotlib.pyplot as plt
import pandas as pd
import os
from gpu_setup import setup_gpu

# initialize the GPU for training (or default to CPU if no GPU is found)
setup_gpu()

from keras._tf_keras.keras import backend as K

# define the directories for the final datasets
final_train_directory = '/app/directory/final_training' #container directory
final_validation_directory = '/app/directory/final_validation'#container directory

# image data generators for rescaling the pixel values
training_data_generator = ImageDataGenerator(rescale=1./255)  # rescale pixel values to [0, 1]
validation_data_generator = ImageDataGenerator(rescale=1./255)

# using flow_from_directory to load images from the training and validation directories
train_generator = training_data_generator.flow_from_directory(
    final_train_directory,
    target_size=(150, 150),  # resizing images to 150x150
    batch_size=64,
    class_mode='binary'  # binary classification (taxiway vs runway)
)

validation_generator = validation_data_generator.flow_from_directory(
    final_validation_directory,
    target_size=(150, 150),  # resizing images to 150x150
    batch_size=64,
    class_mode='binary'  # binary classification
)

# building the CNN model
model = models.Sequential()
model.add(layers.Conv2D(512, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  # output layer for binary classification

# compiling the model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# training the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),  
    epochs=15,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

# Plotting the training and validation accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 6))

# plotting accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# plotting the loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

plot_path = '/app/directory/training_plot.png'
plt.savefig(plot_path)
plt.show()

print(f"Training plot saved at: '/directory/path")
print(f"Training plot saved as: {plot_path}")

# save the model
model.save('/directory/taxiway_v_runway_model.h5')


