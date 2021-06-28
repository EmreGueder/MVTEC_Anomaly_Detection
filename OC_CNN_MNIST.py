import os
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

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


@tf.function
def convert(batch):
    images = tf.expand_dims(batch, axis=-1)
    images = tf.image.resize(images, [224, 224])
    return vgg(tf.image.grayscale_to_rgb(images))


def set_seed(seed):
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


# DEFINE SOME PARAMETERS
SHAPE = (224, 224, 3)
batch_size = 32
set_seed(33)
normal_label = 1
anomaly_label = 2

# Get data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
preprocess_dataset = tf.data.Dataset.from_tensor_slices(train_images).batch(batch_size)

vgg = VGG16(weights='imagenet', include_top=True, input_shape=SHAPE)
vgg = Model(vgg.input, vgg.layers[-3].output)
vgg.trainable = False

prepocesses_batches = []
for idx, batch in enumerate(preprocess_dataset):
    print(f'Processing batch {idx} ...')
    preprocessed_batch = convert(batch).numpy()
    prepocesses_batches.append(preprocessed_batch)

preprocess_data = np.concatenate(prepocesses_batches, axis=0)
print('Success')


# Resize and transform images to RGB
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)
train_images = tf.image.resize(train_images, [224, 224])
test_images = tf.image.resize(test_images, [224, 224])
train_images = tf.image.grayscale_to_rgb(train_images)
test_images = tf.image.grayscale_to_rgb(test_images)
# Get number 'one' from train images
normal_train_images = train_images[train_labels == normal_label]
normal_train_images = tf.data.Dataset.from_tensor_slices(normal_train_images).shuffle(batch_size).batch(batch_size)

normal_test_images = test_images[test_labels == normal_label]
anomaly_test_images = test_images[test_labels == anomaly_label]
valid_images = np.concatenate([normal_test_images, anomaly_test_images], axis=0)
valid_labels = np.concatenate([np.ones(normal_test_images.shape[0]), np.zeros(anomaly_test_images.shape[0])], axis=0)

# Shuffle the test data
rand_idx = np.arange(valid_images.shape[0])
np.random.shuffle(rand_idx)

valid_images = valid_images[rand_idx]
valid_labels = valid_labels[rand_idx]

valid_images = tf.data.Dataset.from_tensor_slices(valid_images)


def get_model(train=True):

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
    model.compile(Adam(lr=1e-4), loss='binary_crossentropy')

    return model


es = EarlyStopping(monitor='val_loss', mode='auto', restore_best_weights=True, verbose=1, patience=5)

model = get_model()
model.fit(normal_train_images, batch_size=batch_size, epochs=20)

valid_labels = np.argmax(np.vstack(valid_labels), axis=1)
print(valid_labels.shape)

# SWITCH TO INFERENCE MODE TO COMPUTE PREDICTIONS
inference_model = get_model(train=False)
inference_model.set_weights(model.get_weights())

# COMPUTE PREDICTIONS ON TEST DATA
pred_test = np.argmax(inference_model.predict(valid_images), axis=1)

# ACCURACY ON TEST DATA
print('ACCURACY:', accuracy_score(valid_labels, pred_test))

# CONFUSION MATRIX ON TEST DATA
cnf_matrix = confusion_matrix(valid_labels, pred_test)

plt.figure(figsize=(7, 7))
plot_confusion_matrix(cnf_matrix, classes=['not ONE', 'ONE'])
plt.show()
