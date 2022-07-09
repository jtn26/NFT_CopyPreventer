import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from keras.applications import densenet

from keras import regularizers
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback

import pathlib
import pickle

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

batch_size = 4
img_height = 631
img_width = 631
epochs = 3


def train_and_valid():
    data_dir = '/home/jtn26/NFT_ML/data'
    data_dir = pathlib.Path(data_dir)

    image_count = len(list(data_dir.glob('*/*.jpg')))

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        label_mode='categorical',
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    label_mode='categorical',
    batch_size=batch_size)

    class_names = train_ds.class_names
    num_classes = len(class_names)

    print(num_classes)
    #AUTOTUNE = tf.data.AUTOTUNE

    #train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    #val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds, image_count, num_classes

def build_model(n_classes):
    base_model = densenet.DenseNet121(input_shape=(img_width, img_height, 3),
                                     weights='imagenet',
                                     include_top=False,
                                     pooling='max')
    for layer in base_model.layers:
      layer.trainable = True
    
    newNet = Sequential()
    newNet.add(keras.layers.Input((img_height, img_width, 3), dtype='float32', name='inputLayer'))
    newNet.add(keras.layers.Rescaling(1./255))
    newNet.add(base_model)
    newNet.add(keras.layers.Dropout(0.2))
    newNet.add(Dense(n_classes, activation='softmax'))
    newNet.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    #print(model.summary())
    return newNet

def get_callback(model):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc', 'mse'])

    early_stop = EarlyStopping(monitor='val_loss', patience=8, verbose=1, min_delta=1e-4)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, min_delta=1e-4)
    callbacks_list = [early_stop, reduce_lr]

    return callbacks_list


def train_model(model, train_ds, valid_ds, callbacks_list):
    model_history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=valid_ds,
        callbacks=callbacks_list)

    return model, model_history


def graphs():
    plt.figure(0)
    plt.plot(model_history.history['acc'],'r')
    plt.plot(model_history.history['val_acc'],'g')
    plt.xticks(np.arange(0, 20, 1.0))
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Num of Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy vs Validation Accuracy")
    plt.legend(['train','validation'])
    plt.savefig("figs.png")

    plt.figure(1)
    plt.plot(model_history.history['loss'],'r')
    plt.plot(model_history.history['val_loss'],'g')
    plt.xticks(np.arange(0, 20, 1.0))
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Num of Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Validation Loss")
    plt.legend(['train','validation'])
    
    plt.show()

    plt.savefig("figs1.png")


def save_to_pickle(model):
    pickle.dump(model, open('model.pkl', 'wb'))


train_ds, valid_ds, image_count, num_classes = train_and_valid()

model = build_model(num_classes)
callbacks_l = get_callback(model)

model.summary()

model, model_history = train_model(model, train_ds, valid_ds, callbacks_l)

save_to_pickle(model)

graphs()

