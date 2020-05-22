import math
import numpy as np
import pandas as pd
import scikitplot
from keras.callbacks import EarlyStopping
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from keras import Sequential
from keras.layers import Dropout, BatchNormalization, Dense, Flatten, MaxPooling2D, Conv2D
from keras.optimizers import Adam
from keras.utils import np_utils

##parameters
batch_size = 32
epochs = 100
learning_rate = 0.001
optim = Adam(learning_rate)
loss = 'categorical_crossentropy'
filename = '/home/eduardo/master/cv/corpora/facial-expression-recognitionferchallenge/fer2013/fer2013/fer2013.csv'

##data importing
df = pd.read_csv(filename)
side = math.sqrt(len(df.pixels[0].split(' ')))
img_array = df.pixels.apply(lambda x: np.array(x.split(' ')).reshape(48, 48, 1).astype('float32'))
img_array = np.stack(img_array, axis=0)

##data preprocessing
le = LabelEncoder()
img_labels = le.fit_transform(df.emotion)
img_labels = np_utils.to_categorical(img_labels)
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

X_train, X_valid, y_train, y_valid = train_test_split(img_array, img_labels,
                                                    shuffle=True, stratify=img_labels,
                                                    test_size=0.1, random_state=12)



img_width = X_train.shape[1]
img_height = X_train.shape[2]
img_depth = X_train.shape[3]
num_classes = y_train.shape[1]

## nn building
net = Sequential(name='CNN')

net.add(
    Conv2D(
        filters=64,
        kernel_size=(5, 5),
        input_shape=(img_width, img_height, img_depth),
        activation='relu',
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
        activation='relu',
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
        activation='relu',
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
        activation='relu',
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
        activation='relu',
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
        activation='relu',
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
        activation='relu',
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
    loss=loss,
    optimizer=optim,
    metrics=['accuracy']
)



net.summary()

#callbacks (not inlcuded in report due to constraints)
es_loss = EarlyStopping()
es_acc = EarlyStopping(monitor='acc')
callbacks = [es_loss]

## nn training
history = net.fit(x=X_train,y=y_train, validation_data=(X_valid,y_valid),epochs=epochs,use_multiprocessing=True, callbacks=callbacks) #steps_per_epoch=len(X_train)/ batch_size

model_yaml = net.to_yaml()
with open("model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)

net.save("model.h5")

# accuracy visualization
yhat_valid = net.predict_classes(X_valid)
scikitplot.metrics.plot_confusion_matrix(np.argmax(y_valid, axis=1), yhat_valid, figsize=(7,7))
pyplot.savefig("confusion_matrix.png")

print(f'total wrong validation predictions: {np.sum(np.argmax(y_valid, axis=1) != yhat_valid)}\n\n')
print(classification_report(np.argmax(y_valid, axis=1), yhat_valid))