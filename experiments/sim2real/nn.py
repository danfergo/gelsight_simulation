import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

from keras.models import Sequential
from keras.layers import Flatten, Dense, BatchNormalization, ReLU, LeakyReLU, Softmax
from keras.applications.resnet_v2 import ResNet50V2


def dense_norm_relu(model, size):
    model.add(Dense(size))
    model.add(BatchNormalization())
    model.add(ReLU())


def nn(n_classes, fc_d=512):
    model = Sequential()

    model.add(ResNet50V2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'))

    model.add(Flatten())
    dense_norm_relu(model, fc_d)
    dense_norm_relu(model, fc_d)

    model.add(Dense(n_classes))
    model.add(Softmax())
    return model
