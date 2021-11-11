import numpy as np
from utils import *

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dropout, MaxPooling2D, Dense, Flatten, LeakyReLU, ReLU
from tensorflow.keras.utils import to_categorical
import random
import cv2

import matplotlib.pyplot as plt
from tqdm import tqdm


def get_dataset_partitions_np(data, labels, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=False):
    assert (train_split + test_split + val_split) == 1

    if shuffle:
        np.random.seed(23)
        randIndx = np.arange(data.shape[0])
        np.random.shuffle(randIndx)
        data = data[randIndx]
        labels = labels[randIndx]

    data_train = data[:int(train_split * len(data))]
    data_valid = data[int(train_split * len(data)):int((train_split + val_split) * len(data))]
    data_test = data[int((train_split + val_split) * len(data)):]

    labels_train = labels[:int(train_split * len(labels))]
    labels_valid = labels[int(train_split * len(labels)):int((train_split + val_split) * len(labels))]
    labels_test = labels[int((train_split + val_split) * len(labels)):]

    data_train = np.array(data_train)
    data_valid = np.array(data_valid)
    data_test = np.array(data_test)
    labels_train = np.array(labels_train)
    labels_valid = np.array(labels_valid)
    labels_test = np.array(labels_test)

    return [data_train, labels_train], [data_valid, labels_valid], [data_test, labels_test]


def create_model():
    activation = LeakyReLU()

    input = Input(shape=(120, 160, 1))

    layer = Conv2D(128, kernel_size=5, strides=1, activation=activation)(input)
    layer = BatchNormalization()(layer)
    layer = MaxPooling2D()(layer)

    layer = Conv2D(128, kernel_size=3, strides=1, activation=activation)(layer)
    layer = BatchNormalization()(layer)
    layer = MaxPooling2D()(layer)

    layer = Conv2D(128, kernel_size=3, strides=1, activation=activation)(layer)
    layer = BatchNormalization()(layer)
    layer = MaxPooling2D()(layer)

    layer = Flatten()(layer)
    layer = Dense(128, activation=activation)(layer)
    layer = Dropout(0.5)(layer)
    output = Dense(4, activation="softmax", name='predictions')(layer)

    model = keras.models.Model(inputs=input, outputs=output)

    opt = keras.optimizers.Nadam(learning_rate=0.000001)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


set_path()

mode = 'depth'
num_channels = 1

data = np.load(f'./data/processed/{mode}_{str(num_channels)}_channel_data.npy')
labels = np.load(f'./data/processed/{mode}_label.npy')
labels = to_categorical(labels)

data = (data.astype(float) - 128) / 128

data_resized = []
for d in data:
    resized = cv2.resize(d, dsize=(160, 120), interpolation=cv2.INTER_LINEAR)
    data_resized.append(resized)

data = np.array(data_resized)

[x_train, y_train], [x_valid, y_valid], [x_test, y_test] = \
    get_dataset_partitions_np(data, labels, shuffle=True)

y_train_labels = np.argmax(y_train, axis=1)
y_valid_labels = np.argmax(y_valid, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

unique, counts = np.unique(y_train_labels, return_counts=True)
print('train label count:', counts)
unique, counts = np.unique(y_valid_labels, return_counts=True)
print('valid label count:', counts)
unique, counts = np.unique(y_test_labels, return_counts=True)
print('test label count:', counts)

# for i in range(24):
#     plt.subplot(4, 6, i+1)
#     plt.imshow(x_train[i])
#
# plt.show()

model = create_model()

early_stopping_cb = keras.callbacks.EarlyStopping(patience=0, monitor='val_loss', restore_best_weights=True)

model.fit(x_train, y_train,
          validation_data=(x_valid, y_valid),
          epochs=5,
          callbacks=[early_stopping_cb])

model.save('./models/first_try')

predictions = model.predict(x_test)
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(test_loss, test_accuracy)

for i in range(24):
    plt.subplot(4, 6, i+1)
    plt.imshow(x_test[i])
    actual_label = np.argmax(y_test[i])
    predicted_label = np.argmax(predictions[i])
    plt.title(f'{actual_label}, {predicted_label}')

plt.show()
