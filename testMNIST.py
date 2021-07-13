import itertools
import os
import random

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import *


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


def set_seed(seed):
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


# DEFINE SOME PARAMETERS
SHAPE = (32, 32, 3)
batch_size = 128
set_seed(33)
normal_label = 1
anomaly_label = 2

# Get data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Resize and transform images to RGB
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)
train_images = tf.image.resize(train_images, [SHAPE[0], SHAPE[1]])
test_images = tf.image.resize(test_images, [SHAPE[0], SHAPE[1]])
train_images = tf.image.grayscale_to_rgb(train_images)
train_images /= 255
test_images = tf.image.grayscale_to_rgb(test_images)

# Get number 'one' from train images
normal_train_images = train_images[train_labels == normal_label]
print(normal_train_images.shape)

normal_train_labels = train_labels[train_labels == normal_label]

# Get test images with number 'one' and 'two'
normal_test_images = test_images[test_labels == normal_label]
anomaly_test_images = test_images[test_labels == anomaly_label]
valid_images = np.concatenate([normal_test_images, anomaly_test_images], axis=0)
valid_labels = np.concatenate([np.ones(normal_test_images.shape[0]), np.zeros(anomaly_test_images.shape[0])], axis=0)

# Shuffle the test data
rand_idx = np.arange(valid_images.shape[0])
np.random.shuffle(rand_idx)

valid_images = valid_images[rand_idx]
valid_labels = valid_labels[rand_idx]


def vgg_feature_extractor(dataset):
    vgg = VGG16(weights='imagenet', include_top=False, input_shape=SHAPE)
    vgg.trainable = False
    vgg_out = vgg.output
    # vgg_out = Flatten()(vgg_out)
    my_vgg = tf.keras.Model(inputs=vgg.input, outputs=vgg_out)
    features = my_vgg.predict(dataset)
    print(features.shape)
    return features


def my_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer((1, 1, 512)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

    return model


def train_step(batch):
    batch_shape = tf.shape(batch)
    noise = tf.random.normal(shape=batch_shape, stddev=0.1)
    new_batch = tf.concat([batch, noise], axis=0)
    new_labels = tf.concat([tf.ones(shape=(batch_shape[0], 1)), tf.zeros(shape=(batch_shape[0], 1))], axis=0)

    with tf.GradientTape() as tape:
        preds = model(new_batch, training=True)
        loss = tf.reduce_sum(tf.keras.losses.binary_crossentropy(new_labels, preds))
        grad = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))

    train_acc_metric.update_state(new_labels, preds)


def train(dataset, epochs):
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        for batch in dataset:
            train_step(batch)

        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))
        train_acc_metric.reset_states()


def get_inference_model(my_model):

    vgg = VGG16(weights='imagenet', include_top=False, input_shape=SHAPE)
    vgg.trainable = False
    vgg_out = vgg.output
    my_inference_model = tf.keras.Model(inputs=vgg.input, outputs=my_model(vgg_out))
    my_inference_model.compile(optimizer=optimizer, loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

    return my_inference_model


model = my_model()
optimizer = tf.keras.optimizers.Adam(lr=1e-4)
# Prepare the metrics.
train_acc_metric = keras.metrics.BinaryAccuracy()

mnist_features = vgg_feature_extractor(normal_train_images)
train_dataset = tf.data.Dataset.from_tensor_slices(mnist_features)
train_dataset = train_dataset.shuffle(batch_size).batch(batch_size)
train(train_dataset, 10)

model.summary()

# SWITCH TO INFERENCE MODE TO COMPUTE PREDICTIONS
inference_model = get_inference_model(model)
inference_model.summary()
print(valid_labels.shape)


# COMPUTE PREDICTIONS ON TEST DATA
print(valid_images.shape)
pred_test = inference_model.predict(valid_images).ravel() < 0.5
fpr_keras, tpr_keras, thresholds_keras = roc_curve(valid_labels, pred_test)
auc_keras = auc(fpr_keras, tpr_keras)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

# # ACCURACY ON TEST DATA
# print('ACCURACY:', accuracy_score(valid_labels, pred_test))
#
# # CONFUSION MATRIX ON TEST DATA
# cnf_matrix = confusion_matrix(valid_labels, pred_test)
#
# plt.figure(figsize=(7, 7))
# plot_confusion_matrix(cnf_matrix, classes=['not ONE', 'ONE'])
# plt.show()
