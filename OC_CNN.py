import os
import random
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import cv2

from sklearn.metrics import confusion_matrix, accuracy_score


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=25)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=15)
    plt.yticks(tick_marks, classes, fontsize=15)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=14)

    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)


# DEFINE SOME PARAMETERS
base_path = "C:/Users/Emre/PycharmProjects/Mnist_CNN_Test/carpet"
SHAPE = (224, 224, 3)
batch_size = 256


def get_patches(images, patch_size):
    all_patches = []
    for rgb_image in images:
        patches = rgb_image.reshape((rgb_image.shape[0] // patch_size,
                                     patch_size, rgb_image.shape[1] // patch_size,
                                     patch_size, 3)).swapaxes(1, 2).reshape((-1, patch_size, patch_size, 3))
        all_patches.extend(patches)

    return all_patches


# GENERATOR WRAPPER TO CREATE FAKE LABEL
def wrap_generator(generator):
    while True:
        x, y = next(generator)
        y = tf.keras.utils.to_categorical(y)
        zeros = tf.zeros_like(y) + tf.constant([1., 0.])
        y = tf.concat([y, zeros], axis=0)

        yield x, y


def set_seed(seed):
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_model(train=True):
    set_seed(33)

    pre_process = Lambda(preprocess_input)
    vgg = VGG16(weights='imagenet', include_top=True, input_shape=SHAPE)
    vgg = Model(vgg.input, vgg.layers[-3].output)
    vgg.trainable = False

    inp = Input(SHAPE)
    vgg_16_process = pre_process(GaussianNoise(0.1)(inp))
    vgg_out = vgg(vgg_16_process)

    noise = Lambda(tf.zeros_like)(vgg_out)
    noise = GaussianNoise(0.1)(noise)

    if train:
        x = Lambda(lambda z: tf.concat(z, axis=0))([vgg_out, noise])
        x = Activation('relu')(x)
    else:
        x = vgg_out

    x = Dense(512, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    out = Dense(2, activation='softmax')(x)

    model = Model(inp, out)
    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy')

    return model


# CREATE EMPTY GENERATORS
train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

# FlOW GENERATORS
train_generator = train_datagen.flow_from_directory(
    base_path + 'train/good/',
    target_size=(SHAPE[0], SHAPE[1]),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    seed=33,
    classes=['normal']
)

test_generator = test_datagen.flow_from_directory(
    base_path + 'test_set/test_set/',
    target_size=(SHAPE[0], SHAPE[1]),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    seed=33,
    classes=['anomaly', 'normal']
)

get_patches = lambda x, y: (tf.reshape(
    tf.image.extract_patches(
        images=tf.expand_dims(x, 0),
        sizes=[1, 3, 3, 1],
        strides=[1, 3, 3, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'), (4, 3, 3, 1)), y)

train_data_patches = tf.data.Dataset.from_tensor_slices(get_patches)


es = EarlyStopping(monitor='val_loss', mode='auto', restore_best_weights=True, verbose=1, patience=5)

model = get_model()
model.fit(wrap_generator(train_generator), steps_per_epoch=train_generator.samples / train_generator.batch_size,
          epochs=20)

# RETRIEVE TEST LABEL FROM GENERATOR
test_num = test_generator.samples

label_test = []
for i in range((test_num // test_generator.batch_size) + 1):
    X, y = test_generator.next()
    label_test.append(y)

label_test = np.argmax(np.vstack(label_test), axis=1)
label_test.shape

# SWITCH TO INFERENCE MODE TO COMPUTE PREDICTIONS
inference_model = get_model(train=False)
inference_model.set_weights(model.get_weights())

# COMPUTE PREDICTIONS ON TEST DATA
pred_test = np.argmax(inference_model.predict(test_generator), axis=1)

# ACCURACY ON TEST DATA
print('ACCURACY:', accuracy_score(label_test, pred_test))

# CONFUSION MATRIX ON TEST DATA
cnf_matrix = confusion_matrix(label_test, pred_test)

plt.figure(figsize=(7, 7))
plot_confusion_matrix(cnf_matrix, classes=['anomaly', 'good'])
plt.show()
