# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 18:17:24 2017
@author: Cédric
"""

import numpy as np
from skimage import color, exposure, transform
from skimage import io
import os
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.cross_validation import train_test_split
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from pathlib import Path
import glob

NUM_CLASSES = 121  # 38
IMG_SIZE = 95
batch_size = 64

d = dict()


def preprocess_img(img):
    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    # roll color axis to axis 0
    img = np.rollaxis(img, -1)
    img = img.reshape(img.shape + (1,))

    return img


def get_class(img_path):
    return img_path.split('\\')[-2]


def cnn_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(IMG_SIZE, IMG_SIZE, 1)))
    model.add(LeakyReLU(alpha=(1 / 3)))
    model.add(Conv2D(16, (3, 3)))
    model.add(LeakyReLU(alpha=(1 / 3)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=(1 / 3)))
    model.add(Conv2D(32, (3, 3)))
    model.add(LeakyReLU(alpha=(1 / 3)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=(1 / 3)))
    model.add(Conv2D(128, (3, 3)))
    model.add(LeakyReLU(alpha=(1 / 3)))
    model.add(Conv2D(64, (3, 3)))
    model.add(LeakyReLU(alpha=(1 / 3)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=(1 / 3)))
    model.add(Conv2D(256, (3, 3)))
    model.add(LeakyReLU(alpha=(1 / 3)))
    model.add(Conv2D(128, (3, 3)))
    model.add(LeakyReLU(alpha=(1 / 3)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=(1 / 3)))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    return model


def prepareData():
    root_dir = 'C:/Users/Baptiste Pouthier/Documents/partage/Dataset/train';
    # root_dir = 'C:/Users/Cédric/Documents/Polytech/MAM5/PFE/uvp5ccelter_group1/uvp5ccelter_group1/';
    imgs = []
    labels = []
    k = 0

    all_img_paths = glob.glob(os.path.join(root_dir, '*/*.jpg'));
    np.random.shuffle(all_img_paths)
    for img_path in all_img_paths:
        img = preprocess_img(io.imread(img_path))
        label = get_class(img_path)
        if (label not in d):
            d[label] = k;
            k = k + 1;
        imgs.append(img)
        labels.append(label)
        if (len(labels) % 100 == 0):
            print(str(len(labels)) + " imgs treated...");
    print(str(len(labels)) + " Done")

    label_chiffre = [];
    for label in labels:
        label_chiffre.append(d.get(label));

    X = np.array(imgs, dtype='float32')
    # Make one hot targets
    Y = np.eye(NUM_CLASSES, dtype='uint8')[label_chiffre]

    return X, Y


def lr_schedule(epoch):
    return lr * (0.1 ** int(epoch / 10))

#images déjà-redimentionnées
#imagesKaggles = 'C:/Users/Cédric/Documents/Polytech/MAM5/PFE/DossierSave/kaggleImage95_95.npy'
#imagesKagglesLabels = 'C:/Users/Cédric/Documents/Polytech/MAM5/PFE/DossierSave/kaggleImageLabels95_95.npy'

#imagesKaggles = 'D:/SaveWeights/Kaggle.npy'
#imagesKagglesLabels = 'D:/SaveWeights/Kaggle_labels.npy'

imagesKaggles = 'C:/Users/Baptiste Pouthier/Documents/partage/Kaggle.npy'
imagesKagglesLabels = 'C:/Users/Baptiste Pouthier/Documents/partage/Kaggle_labels.npy'

# imagesUPV = 'C:/Users/Cédric/Documents/Polytech/MAM5/PFE/DossierSave/upv5Image95_95.npy'
# imagesUPVLabels = 'C:/Users/Cédric/Documents/Polytech/MAM5/PFE/DossierSave/upv5ImageLabels95_95.npy'

pathImages = Path(imagesKaggles);
pathImagesLabels = Path(imagesKagglesLabels);

if (pathImages.exists()):
    print("Load Data...")
    X = np.load(imagesKaggles);
    Y = np.load(imagesKagglesLabels);
else:
    X, Y = prepareData();
    np.save(imagesKaggles, X);
    np.save(imagesKagglesLabels, Y);

X_train, X_val, Y_train, Y_val = train_test_split(X, Y,
                                                  test_size=0.2)  # , random_state=42)

X_val, X_test, Y_val, Y_test = train_test_split(X_val, Y_val, test_size=0.5)

train_datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    # rescale=1./255,
    # zca_whitening=True,
    # zca_epsilon=1e-6,
    rotation_range=45,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    # channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=None  # On pourrai peut être resize ici
    # data_format=K.image_data_format()
)

# test_datagen = ImageDataGenerator(rescale=1./255)

train_datagen.fit(X_train) #entrainement
# test_datagen.fit(X_val)
#
# model = cnn_model()
#
# model.compile(loss='categorical_crossentropy',
#              optimizer='SGD',
#              metrics=['accuracy'])
#
#
#
## Train again
# epochs = 30
# model.fit_generator(train_datagen.flow(X_train, Y_train, batch_size=batch_size),
#                    steps_per_epoch=X_train.shape[0]//batch_size,
#                    epochs=epochs,
#                    validation_data=(X_val,Y_val))
#
##test_datagen.fit(X_test)
# Y_pred = model.predict(X_test)
# acc = np.sum(Y_pred == Y_test) / np.size(Y_pred)
# print("Test accuracy = {}".format(acc))

# test_datagen.fit(X_test)


##☻ METHODE SANS DATA AUGMENTATION
model = cnn_model()

# let's train the model using SGD + momentum
lr = 0.003
sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy',
#              optimizer=sgd,
#              metrics=['accuracy'])

# rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=1e-6)

# weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5

#'D:/SaveWeights/save'
checkpoint = ModelCheckpoint(
    'C:/Users/Baptiste Pouthier/Documents/partage/save',
    monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
model.fit_generator(train_datagen.flow(X_train, Y_train, batch_size=batch_size),
                    steps_per_epoch=X_train.shape[0] // batch_size,
                    epochs=100,
                    validation_data=(X_val, Y_val),
                    callbacks=callbacks_list)  # validation_split pour split auto

# test_datagen = ImageDataGenerator(rescale=1./255)
# test_datagen.fit(X_test)

# model.fit()

Y_pred = model.predict(X_test)

predClasses = Y_pred.argmax(axis=-1)
trueClasses = Y_test.argmax(axis=-1)

acc = np.sum(predClasses == trueClasses) / np.size(predClasses)
print("Test accuracy = {}".format(acc))

# model.summary();0