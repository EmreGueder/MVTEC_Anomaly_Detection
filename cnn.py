# the data, shuffled and split between train and test sets
import tensorflow as tf
from keras.datasets import mnist
from keras.preprocessing import image

# the data, shuffled and split between train and test sets
from tensorflow.python.ops.image_ops_impl import ResizeMethod

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
# print(y_train.shape)

# Resize images to 48x48
x_train = tf.image.resize(x_train, [48, 48], method=ResizeMethod.BILINEAR)
# y_train = tf.image.resize(y_train, [48, 48], method=ResizeMethod.BILINEAR)
# x_test = tf.image.resize(x_test, [48, 48], method=ResizeMethod.BILINEAR)
# y_test = tf.image.resize(y_test, [48, 48], method=ResizeMethod.BILINEAR)

# Convert images to rgb
x_train = tf.image.grayscale_to_rgb(x_train)
# y_train = tf.image.grayscale_to_rgb(y_train)
# x_test = tf.image.grayscale_to_rgb(x_test)
# y_test = tf.image.grayscale_to_rgb(y_test)

print(x_train.shape)
