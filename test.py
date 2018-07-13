from keras.models import load_model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os

import glob
from skimage import io
from skimage import transform

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import sklearn.preprocessing
import pandas as pd

COMMON_PATH = '/home/bpouthie/Documents/partage'
#COMMON_PATH = 'D:/Users/Baptiste Pouthier/Documents/partage'
max_conf = 0.5
graph_archi = nx.Graph()
root_dir = COMMON_PATH+'/clusters/iteration_0/cluster_n0'
IMG_SIZE = 95
batch_size = 64



def get_class(img_path):
    return img_path.split(os.sep)[-2]


def preprocess_img(img,IMG_SIZE):
    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    # roll color axis to axis 0
    img = np.rollaxis(img, -1)
    img = img.reshape(img.shape + (1,))

    return img


def sort_list(list1,list2):
    zipped_pairs = zip(list2,list1)
    sorted_list = [x for _, x in sorted(zipped_pairs)]
    return sorted_list


def prepare_images():
    print("preparation of the images...")
    all_img_paths = glob.glob(os.path.join(root_dir,'*'+os.sep+'*.jpg'))
    images_list=[]
    labels_list=[]
    for img_path in all_img_paths:

        images_list.append(preprocess_img(io.imread(img_path),IMG_SIZE)) #images resized
        labels_list.append(get_class(img_path)) #labels


    d = {ni: indi for indi, ni in enumerate(set(labels_list))}  # assign a number to each unique element in the list "labels", stored in d
    numbers_label = [d[ni] for ni in labels_list]  # list comprehension and store the actual numbers in the numbers_label

    index = np.random.permutation(len(images_list))

    all_images = sort_list(images_list,index)
    all_label_names = sort_list(labels_list,index)
    all_label_numbers = sort_list(numbers_label,index)


    return all_images, all_label_names, all_label_numbers



def divide_images_and_labels(images, labels):

    X_train, X_val, Y_train, Y_val = train_test_split(images, labels, test_size=0.2) #images :all_images     labels: all_label_numbers
    X_val, X_test, Y_val, Y_test = train_test_split(X_val, Y_val, test_size=0.5)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


class architecture_a_plat:
    def __init__(self, NUM_CLASSES, X_train, Y_train, X_test, Y_test, X_val, Y_val, PATH_SAVE_MODEL):
        self.NUM_CLASSES = NUM_CLASSES
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.X_val = X_val
        self.Y_val = Y_val
        self.PATH_SAVE_MODEL = PATH_SAVE_MODEL
        self.batch_size = batch_size
        self.run()


    def cnn_model(self):
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


    def run(self):

        stop_here = EarlyStopping(patience=15)


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


        ##☻ METHODE SANS DATA AUGMENTATION
        model = self.cnn_model()

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
                            epochs=100,
                            #epochs=1,
                            #epochs=50,
                            validation_data=(self.X_val, self.Y_val),
                            callbacks=callbacks_list)  # validation_split pour split auto



        Y_pred = model.predict(self.X_test)

        predClasses = Y_pred.argmax(axis=-1)
        trueClasses = self.Y_test.argmax(axis=-1)

        acc = np.sum(predClasses == trueClasses) / np.size(predClasses)
        print("Test accuracy = {}".format(acc))


print("processing...")

def static_var(varname, value):
    def decorate(func):
        setattr(func, varname, value)
        return func
    return decorate


@static_var("counter", 0)
def ID_static():
    ID_static.counter += 1
    return(ID_static.counter)



def conf_matrix(model,images,labels):

    print("predictions by the model with",len(images),"images")
    Y_prob = model.predict(images)  # belonging probability of images
    Y_classes = Y_prob.argmax(axis=1)  # label predicts the image based on its probability

    true_label = []
    for row in range(0, len(images)):
        true_label.append(np.where(labels[row][np.newaxis, :] == 1)[1][0])  # theoretical label

    conf = confusion_matrix(true_label,Y_classes)

    conf = sklearn.preprocessing.normalize(conf, axis=1, norm='l1')
    #the confusion matrix is now normalized.
    #with row: how each class has been classified, i.e. C[i,j] / sum(conf[i]) of the class i objects are classified as class j
    #with column: what classes are  responsible for each classification, i.e. C[i,j] / sum(conf[j]) of the objects classified as class j were from class i

    return conf



def create_graph(conf,label_dict): #label_dict indicate the true label
    g = nx.Graph()  # graph creation
    above_max = np.where(conf > max_conf) #test against the threshold

    for row, col in zip(above_max[0],above_max[1]):
        if row != col:
            if bool(label_dict): g.add_edge(label_dict[row],label_dict[col]) #create graph
            else : g.add_edge(row,col)
    return g



def from_graph_to_clusters(g): #ajouter nb_iter
    sub_graphs = nx.connected_component_subgraphs(g) #sub_graphs contains the different connected structures

    clusters = [] #will contain the different clusters (numbers)

    ID = []

    for index, sg in enumerate(sub_graphs):
        clusters.append(list(sg.nodes)) #cluster from the subgraphs
        ID.append(ID_static())


    return clusters, ID


def cluster_training(clusters,ID,nb_iter,all_images,all_label_numbers,all_label_names):
    if clusters:

        print("Start Training\n")
        ID_index = -1
        for cluster in clusters:

            ID_index += 1

            print("cluster number", cluster)
            # NUM_CLASSES

            NUM_CLASSES = len(cluster)
            print('number of species: ',NUM_CLASSES)


            cluster_index = [idx for idx, element in enumerate(all_label_numbers) if element in cluster] #give all indexes in all_label_numbers where the label is in cluster

            cluster_images =[]
            cluster_labels = []
            cluster_names = []#

            for index in cluster_index: #define the custer images and the labels

                cluster_images.append(all_images[index])
                cluster_labels.append(all_label_numbers[index])
                cluster_names.append(all_label_names[index])#

            print('cluster composed by', set(cluster_names))
            print('image number:',len(cluster_labels),"\n")


            #### transform labels in adapted one-hot encoding ####
            unique = list(set(cluster_labels))
            new_values = list(range(0, len(unique)))

            label_dict = dict()
            if len(cluster_labels) >= 2*batch_size:

                rectified_labels = []

                for element in cluster_labels:
                    for value in unique:
                        if element == value: rectified_labels.append(new_values[unique.index(value)])
                        label_dict[new_values[unique.index(value)]] = value

            ####

                if not label_dict in all_clusters_with_ID[3::4]:


                    X_train, Y_train, X_val, Y_val, X_test, Y_test = divide_images_and_labels(np.array(cluster_images,dtype='float32'), np.eye(NUM_CLASSES, dtype='uint8')[rectified_labels]) #cluster_labels


                    all_clusters_with_ID.extend((ID[ID_index],X_test,Y_test,label_dict))#save the images and label for future call in confusion_matrix


                    # PATH_SAVE_MODEL
                    PATH_SAVE_ITER_DIR = COMMON_PATH+'/clusters_saves/iteration_'+str(nb_iter)
                    if not os.path.isdir(PATH_SAVE_ITER_DIR):
                        os.mkdir(PATH_SAVE_ITER_DIR)
                    PATH_SAVE_MODEL_DIR = COMMON_PATH+'/clusters_saves/iteration_'+str(nb_iter)+'/save_cluster_'+str(ID[ID_index])
                    os.mkdir(PATH_SAVE_MODEL_DIR)
                    PATH_SAVE_MODEL = PATH_SAVE_MODEL_DIR+'/save'

                    df = pd.DataFrame(list(set(cluster_names))) #save the species name regarding each cluster to help the final prediction afterward
                    df.to_csv(PATH_SAVE_MODEL_DIR+'/labels.csv', index=False) #to load afterward : my_data = np.genfromtxt(PATH_SAVE_MODEL_DIR+'labels.csv', dtype=None,encoding=None)[1:]


                    architecture_a_plat(NUM_CLASSES, X_train, Y_train, X_test, Y_test, X_val, Y_val, PATH_SAVE_MODEL)
                    print("\n-cluster trained-\n")

                else : print(" /!\ cluster already present at the previous iteration, no training  /!\ ")

            else : print(" /!\ too few images, no training /!\ \n\n ")

    else: print("no more cluster - no training needed")

    return(all_clusters_with_ID)





# all_images = np.load(COMMON_PATH+'/all_images.npy')
# all_label_names = np.load(COMMON_PATH+'/all_label_names.npy')
# all_label_numbers = np.load(COMMON_PATH+'/all_label_numbers.npy')


#decommenter les trois lignes ci-dessous pour lancer le test à 0 (partir de 121 classes) /!\ necessaire pour bon resultats car le "save" actuel n'est pas adapté au melange fait.
all_images, all_label_names, all_label_numbers= prepare_images()

# save(this is not mandatory)
np.save(COMMON_PATH + "/all_images", all_images)
np.save(COMMON_PATH + "/all_label_names", all_label_names)
np.save(COMMON_PATH + "/all_label_numbers", all_label_numbers)

X_train, Y_train, X_val, Y_val, X_test, Y_test = divide_images_and_labels(np.array(all_images,dtype='float32'), np.eye(121, dtype='uint8')[all_label_numbers])

# save(this is not mandatory)
np.save(COMMON_PATH +"/X_test", X_test)
np.save(COMMON_PATH +"/Y_test", Y_test)

architecture_a_plat(121, X_train, Y_train, X_test, Y_test, X_val, Y_val, COMMON_PATH+'/clusters_saves/iteration_0/save_cluster_0/save')


#commenter les deux lignes ci dessous pour lancer le test à 0
# X_test = np.load(COMMON_PATH+'/X_test.npy')
# Y_test = np.load(COMMON_PATH+'/Y_test.npy')

empty_dict=dict()
all_clusters_with_ID=[0,X_test,Y_test,empty_dict]
clusters = True
iteration = 0

while(clusters):

    iteration += 1
    clusters_for_an_iteration = COMMON_PATH+'/clusters_saves/iteration_'+str(iteration-1)

    if os.path.isdir(clusters_for_an_iteration):

        print("-------------------------------------------------------------------------------")
        print("--------------------------------- iteration", iteration - 1, "---------------------------------")
        print("-------------------------------------------------------------------------------")

        for cluster in list(os.listdir(clusters_for_an_iteration)):
            print("\ncluster :",cluster)

            nb_cluster = int(cluster.split('_')[-1])


            X_test, Y_test = all_clusters_with_ID[all_clusters_with_ID[::4].index(nb_cluster)*4+1],all_clusters_with_ID[all_clusters_with_ID[::4].index(nb_cluster)*4+2]

            dict_references=all_clusters_with_ID[all_clusters_with_ID[::4].index(nb_cluster)*4+3]


            model = load_model(COMMON_PATH+'/clusters_saves/iteration_'+str(iteration-1)+'/'+cluster+'/save')


            confus_matrix = conf_matrix(model, X_test, Y_test)
            # plt.matshow(confus_matrix)
            # plt.show()

            G = create_graph(confus_matrix,dict_references)
            # nx.draw_networkx(G)
            # plt.show()

            clusters, ID = from_graph_to_clusters(G)

            #graph_archi.add_node(cluster,pos=(nb_cluster,iteration-1))

            # for node in list_nodes_graph:
            #     #graph_archi.add_node(node,pos=((node.split('_')[-2]).split('n')[-1],iteration))
            #     #graph_archi.add_edge(node,cluster)

            all_clusters_with_ID = cluster_training(clusters,ID,iteration,all_images,all_label_numbers,all_label_names)

    else :
        print("\nThere are only redundant groups")
        clusters = False

print("training finished!")
# pos = nx.get_node_attributes(graph_archi,'pos')
# nx.draw_networkx(graph_archi,pos)
# plt.show()






