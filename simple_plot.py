import tensorflow as tf

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

resize = lambda x, y: (tf.image.resize(tf.expand_dims(x, -1), (224, 224)), y)

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).map(resize)

for image, label in train_ds.take(5):
    print(image.shape)
