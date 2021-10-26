import os
import pathlib
import random

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE


def set_seed(seed):
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


# Define some parameters
train_data_dir = 'dataset/tile/train/'
test_data_dir = 'dataset/tile/test/'
ground_truth_data_dir = 'dataset/tile/ground_truth/'
batch_size = 32
set_seed(33)
patch_size = 32
img_height = 512
img_width = 512


def get_labels(dataset):
    labels = []
    for image in dataset:
        if np.any(image > 0):
            labels.append(1)
        else:
            labels.append(0)
    labels = np.asarray(labels)
    return labels


def remove_excessive_normal_images(test_dataset, ground_truth_dataset, labels):
    my_list = []
    counter = 0
    anomaly_count = 0
    for image in ground_truth_dataset:
        if np.any(image > 0):
            anomaly_count += 1

    for image in ground_truth_dataset:
        if np.all(image == 0):
            my_list.append(counter)
        counter += 1
        if len(my_list) >= ground_truth_dataset.shape[0] - 2 * anomaly_count:
            break
    new_dataset = np.delete(test_dataset, my_list, axis=0)
    new_labels = np.delete(labels, my_list, axis=0)
    return new_dataset, new_labels


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

def get_inference_model(my_model):
    vgg = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    vgg.trainable = False
    vgg_out = vgg.output
    my_inference_model = tf.keras.Model(inputs=vgg.input, outputs=my_model(vgg_out))

    return my_inference_model

model = my_model()

train_ds = tf.data.Dataset.list_files(str(pathlib.Path(train_data_dir + '*.png')))

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
train_ds_patches = get_patches(train_ds, patch_size)
train_ds_patches = np.asarray(train_ds_patches)
train_labels = tf.zeros(shape=(train_ds_patches.shape[0]))
train_labels = np.asarray(train_labels)
print("Shape of train dataset: ", train_ds_patches.shape)
print("Shape of train labels: ", train_labels.shape)
print("Shape of train labels: ", train_labels)

# Get the ground truth dataset for labeling the images/patches
ground_truth_ds = tf.data.Dataset.list_files(str(pathlib.Path(ground_truth_data_dir + '*.png')), shuffle=False)
for f in ground_truth_ds.take(10):
    print(f.numpy())
ground_truth_ds = ground_truth_ds.map(process_path, num_parallel_calls=AUTOTUNE)
ground_truth_ds = get_patches(ground_truth_ds, patch_size)
ground_truth_ds = np.asarray(ground_truth_ds)
print("Shape of ground truth dataset: ", ground_truth_ds.shape)

test_labels = get_labels(ground_truth_ds)
print("Shape of test labels: ", test_labels.shape)

# Get the test dataset
test_ds = tf.data.Dataset.list_files(str(pathlib.Path(test_data_dir + '*.png')), shuffle=False)
for f in test_ds.take(10):
    print(f.numpy())
test_ds = test_ds.map(process_path, num_parallel_calls=AUTOTUNE)
test_ds = get_patches(test_ds, patch_size)
test_ds = np.asarray(test_ds)
print("Shape of test dataset", test_ds.shape)
test_ds, test_labels = remove_excessive_normal_images(test_ds, ground_truth_ds, test_labels)
print("Shape of new test dataset with removed normal pictures:", test_ds.shape)
print("Shape of new test labels with removed normal pictures:", test_labels.shape)

test_ds, trainA_ds, test_labels, trainA_labels = train_test_split(test_ds, test_labels, test_size=0.6, stratify=test_labels)
print("Shape of split test dataset:", test_ds.shape)
print("Shape of split test labels:", test_labels.shape)
print("Shape of train anomalies dataset:", trainA_ds.shape)
print("Shape of train anomalies labels:", trainA_labels.shape)
train_ds_patches = np.concatenate((train_ds_patches, trainA_ds), axis=0)
train_labels = np.concatenate((train_labels, trainA_labels), axis=0)
print("Shape of train dataset with anomalies:", train_ds_patches.shape)
print("Shape of train labels with anomalies:", train_labels.shape)

# Split a validation subset from the test dataset
test_ds, valid_ds, test_labels, valid_labels = train_test_split(test_ds, test_labels, test_size=0.5, stratify=test_labels)
print("Shape of validation dataset:", valid_ds.shape)
print("Shape of validation labels:", valid_labels.shape)
print("Shape of split test dataset:", test_ds.shape)
print("Shape of split test labels:", test_labels.shape)

train_ds_features = vgg_feature_extractor(train_ds_patches)
train_ds_patches = tf.data.Dataset.from_tensor_slices((train_ds_features, train_labels))
train_ds_patches = configure_for_performance(train_ds_patches)

valid_ds_features = vgg_feature_extractor(valid_ds)
valid_ds = tf.data.Dataset.from_tensor_slices((valid_ds_features, valid_labels))
valid_ds = configure_for_performance(valid_ds)

model.compile(optimizer='adam', loss=keras.losses.binary_crossentropy, metrics=['accuracy'])
model.summary()
results = model.fit(train_ds_patches, epochs=3, validation_data=valid_ds)

# Switch to inference mode to compute predictions
inference_model = get_inference_model(model)
inference_model.summary()

# Compute predictions on test data
pred_test = inference_model.predict(test_ds).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(test_labels, pred_test, pos_label=1)
auc_keras = auc(fpr_keras, tpr_keras)


plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='VGG-16 (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve VGG-16, Kategorie Fliese')
plt.legend(loc='best')
plt.show()





