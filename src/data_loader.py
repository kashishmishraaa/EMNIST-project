import tensorflow as tf
from tensorflow.keras.utils import to_categorical

def load_emnist_dataset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()  # Replace with EMNIST loader
    x_train = tf.image.resize_with_pad(tf.expand_dims(x_train, -1), 64, 64) / 255.0
    x_test = tf.image.resize_with_pad(tf.expand_dims(x_test, -1), 64, 64) / 255.0
    y_train = to_categorical(y_train, 47)
    y_test = to_categorical(y_test, 47)
    return x_train, y_train, x_test, y_test
