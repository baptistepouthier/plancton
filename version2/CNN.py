from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np



class cnn:
    def __init__(self, NUM_CLASSES, X_train, Y_train, X_test, Y_test, X_val, Y_val, PATH_SAVE_MODEL,batch_size,img_size,model):
        self.NUM_CLASSES = NUM_CLASSES
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.X_val = X_val
        self.Y_val = Y_val
        self.PATH_SAVE_MODEL = PATH_SAVE_MODEL
        self.batch_size = batch_size
        self.IMG_SIZE = img_size
        self.previous_model = model
        self.run()



    def cnn_model(self):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), padding='same',
                             input_shape=(self.IMG_SIZE, self.IMG_SIZE, 1)))
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
        model.add(Conv2D(128, (3, 3)))
        model.add(LeakyReLU(alpha=(1 / 3)))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=(1 / 3)))
        model.add(Dropout(0.5))
        model.add(Dense(self.NUM_CLASSES, activation='softmax'))
        return model


    # def transfer_cnn(self):
    #
    #     self.previous_model.pop()
    #     model = self.previous_model.add(Dense(self.NUM_CLASSES, activation='softmax'))
    #
    #     return model


    def run(self):

        print('total number of images used for training:', self.X_train.shape[0] * self.batch_size)

        stop_here = EarlyStopping(patience=10)



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

        train_datagen.fit(np.array(self.X_train))

        # print(self.previous_model)

        if self.previous_model is None:
            model = self.cnn_model()

        else :
            print("using weight from previous model")

            model = load_model(self.previous_model)
            model.pop()
            model.add(Dense(self.NUM_CLASSES, activation='softmax'))
            model.layers[-1].name = 'dense_2'


        # let's train the model using SGD + momentum
        lr = 0.003
        sgd = SGD(lr=lr, momentum=0.9, nesterov=True)



        checkpoint = ModelCheckpoint(
            self.PATH_SAVE_MODEL,
            monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint,stop_here]
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
        model.fit_generator(train_datagen.flow(self.X_train,self.Y_train, batch_size=self.batch_size),
                            steps_per_epoch=self.X_train.shape[0] // self.batch_size,
                            #epochs=100,
                            # epochs=1,
                            epochs=50,
                            validation_data=(self.X_val, self.Y_val),
                            callbacks=callbacks_list)  # validation_split pour split auto



        Y_pred = model.predict(self.X_test)

        predClasses = Y_pred.argmax(axis=-1)
        trueClasses = self.Y_test.argmax(axis=-1)

        acc = np.sum(predClasses == trueClasses) / np.size(predClasses)
        print("Test accuracy = {}".format(acc))



