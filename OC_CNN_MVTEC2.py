import os
import pathlib
import random

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.applications import VGG16
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE


def set_seed(seed):
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


# DEFINE SOME PARAMETERS
train_data_dir = 'carpet/train/'
test_data_dir = 'carpet/test/'
ground_truth_data_dir = 'carpet/ground_truth/'
batch_size = 32
set_seed(33)
patch_size = 32
img_height = 512
img_width = 512


def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)

    label = 0
    return label


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_png(img, channels=3)
    # resize the image to the desired size
    return tf.image.resize(img, [img_height, img_width])


def process_path(file_path):
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def get_patches(images, patch_size):
    all_patches = []
    for rgb_image in images:
        rgb_image = rgb_image.numpy()
        patches = rgb_image.reshape((rgb_image.shape[0] // patch_size,
                                     patch_size, rgb_image.shape[1] // patch_size,
                                     patch_size, 3)).swapaxes(1, 2).reshape((-1, patch_size, patch_size, 3))
        all_patches.extend(patches)

    return all_patches


def vgg_feature_extractor(dataset):
    vgg = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    vgg.trainable = False
    vgg_out = vgg.output
    my_vgg = tf.keras.Model(inputs=vgg.input, outputs=vgg_out)
    features = my_vgg.predict(dataset)
    print("Shape of extracted features: ", features.shape)
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
    new_labels = tf.concat([tf.zeros(shape=(batch_shape[0], 1)), tf.ones(shape=(batch_shape[0], 1))], axis=0)

    with tf.GradientTape() as tape:
        preds = model(new_batch, training=True)
        loss = tf.reduce_sum(tf.keras.losses.binary_crossentropy(new_labels, preds))
        grad = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grad, model.trainable_variables))

    train_acc_metric.update_state(new_labels, preds)
    train_loss_metric.update_state(new_labels, preds)


def train(dataset, epochs):
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        for batch in dataset:
            train_step(batch)

        train_acc = train_acc_metric.result()
        train_loss = train_loss_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))
        print("Training loss over epoch: %.4f" % (float(train_loss),))
        train_acc_metric.reset_states()


def get_inference_model(my_model):
    vgg = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    vgg.trainable = False
    vgg_out = vgg.output
    my_inference_model = tf.keras.Model(inputs=vgg.input, outputs=my_model(vgg_out))
    my_inference_model.compile(optimizer=optimizer, loss=keras.losses.binary_crossentropy, metrics=['accuracy'])

    return my_inference_model


model = my_model()
optimizer = tf.keras.optimizers.Adam(lr=1e-4)
# Prepare the metrics.
train_acc_metric = keras.metrics.BinaryAccuracy()
train_loss_metric = keras.metrics.BinaryCrossentropy()

train_ds = tf.data.Dataset.list_files(str(pathlib.Path(train_data_dir + '*.png')))

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
train_ds_patches = get_patches(train_ds, patch_size)
train_ds_patches = np.asarray(train_ds_patches)

ground_truth_ds = tf.data.Dataset.list_files(str(pathlib.Path(ground_truth_data_dir + '*.png')), shuffle=False)
for f in ground_truth_ds.take(10):
    print(f.numpy())
ground_truth_ds = ground_truth_ds.map(process_path, num_parallel_calls=AUTOTUNE)
ground_truth_ds = get_patches(ground_truth_ds, patch_size)
ground_truth_ds = np.asarray(ground_truth_ds)
print(ground_truth_ds.shape)

test_labels = []

for image in ground_truth_ds:
    if np.any(image > 0):
        test_labels.append(1)
    else:
        test_labels.append(0)

test_labels = np.asarray(test_labels)
print(test_labels)

test_ds = tf.data.Dataset.list_files(str(pathlib.Path(test_data_dir + '*.png')), shuffle=False)
for f in test_ds.take(10):
    print(f.numpy())
test_ds = test_ds.map(process_path, num_parallel_calls=AUTOTUNE)
test_ds = get_patches(test_ds, patch_size)
test_ds = np.asarray(test_ds)
print(test_ds.shape)

train_ds_features = vgg_feature_extractor(train_ds_patches)
train_ds_patches = tf.data.Dataset.from_tensor_slices(train_ds_features)
train_ds_patches = configure_for_performance(train_ds_patches)
train(train_ds_patches, 3)

model.summary()

# SWITCH TO INFERENCE MODE TO COMPUTE PREDICTIONS
inference_model = get_inference_model(model)
inference_model.summary()
print("Shape of test labels: ", test_labels.shape)

# COMPUTE PREDICTIONS ON TEST DATA
print("Shape of test dataset: ", test_ds.shape)
pred_test = inference_model.predict(test_ds).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_labels, pred_test, pos_label=1)
auc_keras = auc(fpr_keras, tpr_keras)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='OC_CNN (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

# Zoom in view of the upper left corner.
plt.figure(2)
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='OC_CNN (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (zoomed in at top left)')
plt.legend(loc='best')
plt.show()
