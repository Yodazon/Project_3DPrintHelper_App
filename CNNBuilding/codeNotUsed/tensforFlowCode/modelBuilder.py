#Preprocessing Images for tensorflow CNN model
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from sklearn.model_selection import train_test_split

#Define dimensions for images
img_height = 227
img_width = 227
batch_size = 50

img_dir = "C:\\Coding\\Github\\Project_3DPrintingApp\\CNNBuilding\\images_combined"


train_ds = tf.keras.utils.image_dataset_from_directory(
    img_dir,
    validation_split = 0.2,
    subset ="training",
    seed = 342,
    image_size = (img_height, img_width),
    batch_size = batch_size
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    img_dir,
    validation_split = 0.2,
    subset ="validation",
    seed = 342,
    image_size = (img_height, img_width),
    batch_size = batch_size
)

#Defining tensor flow model
model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(4)
])

#configure for performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.optimizers.SGD(learning_rate=0.001),
    metrics=['accuracy']    
)
# print(tf.config.list_physical_devices('GPU').encode('utf-8').decode('cp1252'))


# Step 4: Train your model
history = model.fit(train_ds,
                    batch_size=50,
                    shuffle=True, 
                    epochs=10, 
                    validation_data=val_ds)