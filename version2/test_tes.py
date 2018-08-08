from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.models import load_model
import os

# model = load_model('D:/Users/Baptiste Pouthier/Documents/partage/clusters_saves/iteration_0/save_cluster_0/save')
#
# print(model.summary())
#
# model.add(Dense(2,activation='softmax'))
#
#
# print(model.summary())

model_path = 'D:/Users/Baptiste Pouthier/Documents/partage/clusters_saves/iteration_0/save_cluster_0/save'

print(model_path.split('/')[-2].split('_')[-1])
