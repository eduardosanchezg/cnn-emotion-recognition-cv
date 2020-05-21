import math
import numpy as np
import pandas as pd
#from __future__ import absolute_import, division, print_function, unicode_literals

import scikitplot
import seaborn as sns
from matplotlib import pyplot


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

import tensorflow as tf
from keras import Sequential
from keras.datasets import mnist
from keras.layers import Dropout, BatchNormalization, LeakyReLU, Activation, Dense, Flatten, MaxPooling2D, Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import adam, Adam

from keras.utils import np_utils

optim = Adam(0.001)

df = pd.read_csv('/home/eduardo/master/cv/corpora/facial-expression-recognitionferchallenge/fer2013/fer2013/fer2013.csv')

#print(df.shape)
#print(df.head())

#print(df.emotion.unique())

emotion_label_to_text = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral'}

#print(df.Usage.value_counts())

side = math.sqrt(len(df.pixels[0].split(' ')))
#
# fig = pyplot.figure(1, (14, 14))
#
# k = 0
# for label in sorted(df.emotion.unique()):
#     for j in range(7):
#         px = df[df.emotion==label].pixels.iloc[k]
#         px = np.array(px.split(' ')).reshape(48, 48).astype('float32')
#
#         k += 1
#         ax = pyplot.subplot(7, 7, k)
#         ax.imshow(px, cmap='gray')
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.set_title(emotion_label_to_text[label])
#         pyplot.tight_layout()
#
# pyplot.show()

img_array = df.pixels.apply(lambda x: np.array(x.split(' ')).reshape(48, 48, 1).astype('float32'))
img_array = np.stack(img_array, axis=0)

le = LabelEncoder()
img_labels = le.fit_transform(df.emotion)
img_labels = np_utils.to_categorical(img_labels)


le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
#print(le_name_mapping)

X_train, X_valid, y_train, y_valid = train_test_split(img_array, img_labels,
                                                    shuffle=True, stratify=img_labels,
                                                    test_size=0.1, random_state=42)

print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)

img_width = X_train.shape[1]
img_height = X_train.shape[2]
img_depth = X_train.shape[3]
num_classes = y_train.shape[1]

X_train = X_train / 255
X_valid = X_valid / 255

net = Sequential(name='DCNN')

net.add(
    Conv2D(
        filters=64,
        kernel_size=(5, 5),
        input_shape=(img_width, img_height, img_depth),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_1'
    )
)
net.add(BatchNormalization(name='batchnorm_1'))
net.add(
    Conv2D(
        filters=64,
        kernel_size=(5, 5),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_2'
    )
)
net.add(BatchNormalization(name='batchnorm_2'))

net.add(MaxPooling2D(pool_size=(2, 2), name='maxpool2d_1'))
net.add(Dropout(0.4, name='dropout_1'))

net.add(
    Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_3'
    )
)
net.add(BatchNormalization(name='batchnorm_3'))
net.add(
    Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_4'
    )
)
net.add(BatchNormalization(name='batchnorm_4'))

net.add(MaxPooling2D(pool_size=(2, 2), name='maxpool2d_2'))
net.add(Dropout(0.4, name='dropout_2'))

net.add(
    Conv2D(
        filters=256,
        kernel_size=(3, 3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_5'
    )
)
net.add(BatchNormalization(name='batchnorm_5'))
net.add(
    Conv2D(
        filters=256,
        kernel_size=(3, 3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_6'
    )
)
net.add(BatchNormalization(name='batchnorm_6'))

net.add(MaxPooling2D(pool_size=(2, 2), name='maxpool2d_3'))
net.add(Dropout(0.5, name='dropout_3'))

net.add(Flatten(name='flatten'))

net.add(
    Dense(
        128,
        activation='elu',
        kernel_initializer='he_normal',
        name='dense_1'
    )
)
net.add(BatchNormalization(name='batchnorm_7'))

net.add(Dropout(0.6, name='dropout_4'))

net.add(
    Dense(
        num_classes,
        activation='softmax',
        name='out_layer'
    )
)

net.compile(
    loss='categorical_crossentropy',
    optimizer=optim,
    metrics=['accuracy']
)

batch_size = 32 #batch size of 32 performs the best.
epochs = 100

history = net.fit(x=X_train,y=y_train, validation_data=(X_valid,y_valid), steps_per_epoch=len(X_train)/ batch_size,epochs=epochs,use_multiprocessing=True)

model_yaml = net.to_yaml()
with open("model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)

net.save("model.h5")

yhat_valid = net.predict_classes(X_valid)
scikitplot.metrics.plot_confusion_matrix(np.argmax(y_valid, axis=1), yhat_valid, figsize=(7,7))
pyplot.savefig("confusion_matrix_dcnn.png")

print(f'total wrong validation predictions: {np.sum(np.argmax(y_valid, axis=1) != yhat_valid)}\n\n')
print(classification_report(np.argmax(y_valid, axis=1), yhat_valid))