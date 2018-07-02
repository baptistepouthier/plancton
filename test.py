from keras.models import load_model
#import numpy as np
#import matplotlib.pyplot as plt
import networkx as nx
import os
#import pyprind
import distutils.dir_util
#from confusion import architecture_a_plat as cnn
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
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from pathlib import Path
import glob


class architecture_a_plat:
    def __init__(self, NUM_CLASSES, PATH_IMAGES, PATH_LABELS, PATH_SAVE_MODEL, PATH_PREPARED_IMAGES, PATH_TEST_IMAGES,PATH_TEST_LABELS,PATH_LABEL_DICT):

        self.NUM_CLASSES = NUM_CLASSES
        self.PATH_IMAGES = PATH_IMAGES
        self.PATH_LABELS = PATH_LABELS
        self.PATH_SAVE_MODEL = PATH_SAVE_MODEL
        self.PATH_PREPARED_IMAGES = PATH_PREPARED_IMAGES

        self.PATH_LABEL_DICT = PATH_LABEL_DICT

        self.PATH_TEST_IMAGES = PATH_TEST_IMAGES
        self.PATH_TEST_LABELS = PATH_TEST_LABELS


        self.IMG_SIZE=95
        self.batch_size=32
        self.d=dict()
        self.run()


    def preprocess_img(self, img):
        # rescale to standard size
        img = transform.resize(img, (self.IMG_SIZE, self.IMG_SIZE))

        # roll color axis to axis 0
        img = np.rollaxis(img, -1)
        img = img.reshape(img.shape + (1,))

        return img


    def get_class(self, img_path):
        return img_path.split('/')[-2]


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
        model.add(LeakyReLU(alpha=(1 / 3)))
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


    def prepareData(self):
        root_dir = self.PATH_IMAGES

        imgs = []
        labels = []
        k = 0

        all_img_paths = glob.glob(os.path.join(root_dir, '*/*.jpg'))
        np.random.shuffle(all_img_paths)
        for img_path in all_img_paths:
            img = self.preprocess_img(io.imread(img_path))
            label = self.get_class(img_path)
            if (label not in self.d):
                self.d[label] = k
                k = k + 1
            imgs.append(img)
            labels.append(label)
            if (len(labels) % 100 == 0):
                print(str(len(labels)) + " imgs treated...")
        print(str(len(labels)) + " Done")
        # np.save(self.PATH_K, k) #save number of classes
        # np.save(self.PATH_TEST_LABELS_NAMES, labels) #save labels names sorted
        label_chiffre = []
        for label in labels:
            label_chiffre.append(self.d.get(label))
        dictlist = []
        for key, value in self.d.items():
            temp = [key, value]
            dictlist.append(temp)

        np.save(self.PATH_LABEL_DICT,dictlist) #save the dictionary

        X = np.array(imgs, dtype='float32')
        # Make one hot targets
        Y = np.eye(self.NUM_CLASSES, dtype='uint8')[label_chiffre]

        return X, Y


    #def lr_schedule(self, epoch): //pas sur de commenter ça
    #    return lr * (0.1 ** int(epoch / 10))

    #images déjà-redimentionnées
    #imagesKaggles = 'C:/Users/Cédric/Documents/Polytech/MAM5/PFE/DossierSave/kaggleImage95_95.npy'
    #imagesKagglesLabels = 'C:/Users/Cédric/Documents/Polytech/MAM5/PFE/DossierSave/kaggleImageLabels95_95.npy'


    def run(self):

        imagesKaggles = self.PATH_PREPARED_IMAGES
        imagesKagglesLabels = self.PATH_LABELS

        stop_here = EarlyStopping(patience=2)
        # imagesUPV = 'C:/Users/Cédric/Documents/Polytech/MAM5/PFE/DossierSave/upv5Image95_95.npy'
        # imagesUPVLabels = 'C:/Users/Cédric/Documents/Polytech/MAM5/PFE/DossierSave/upv5ImageLabels95_95.npy'

        pathImages = Path(imagesKaggles)
        pathImagesLabels = Path(imagesKagglesLabels)

        if (pathImages.exists()):
            print("Load Data...")
            X = np.load(imagesKaggles)
            Y = np.load(imagesKagglesLabels)
        else:
            X, Y = self.prepareData()
            np.save(imagesKaggles, X)
            np.save(imagesKagglesLabels, Y)

        X_train, X_val, Y_train, Y_val = train_test_split(X, Y,
                                                          test_size=0.2)  # , random_state=42)

        X_val, X_test, Y_val, Y_test = train_test_split(X_val, Y_val, test_size=0.5)

        print("save test images...")
        np.save(self.PATH_TEST_IMAGES,X_test)

        print("save test labels...")
        np.save(self.PATH_TEST_LABELS,Y_test)

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
        model = self.cnn_model()

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
            self.PATH_SAVE_MODEL,
            monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint,stop_here]
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
        model.fit_generator(train_datagen.flow(X_train, Y_train, batch_size=self.batch_size),
                            steps_per_epoch=X_train.shape[0] // self.batch_size,
                            epochs=100,
                            #epochs=1,
                            #epochs=50,
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





#COMMON_PATH = 'D:/Users/Baptiste Pouthier/Documents/partage'
COMMON_PATH = '/home/bpouthie/Documents/partage'
max_conf = 0.7
graph_archi = nx.Graph()


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


def get_number_images(images):
    # #list_dir = os.listdir(dataset)
    # list_dir = np.load(sorted_labels)
    # #number_images = sum(len(files) for _, _, files in os.walk(dataset))
    number_images = len(images)
    return number_images


def dict_information(label_dict):
    sorted_labels = []

    for label in label_dict:
        sorted_labels.append(label[0])

    return len(sorted_labels)



def confusion_matrix(number_classes,number_images,model,images,LABELS):

    #number_classes = len(list_directory) ###########################
    print("number of classes :", number_classes)
    print("number of images :", number_images)
    conf = np.zeros((number_classes, number_classes))  # confusion matrix creation
    print("\n","predictions by the model:")

    #for row in pyprind.prog_percent(range(0,number_images)):
    for row in range(0, number_images):

        Y_prob = model.predict(images[row][np.newaxis, :]) #belonging probability of images

        Y_classes = Y_prob.argmax(axis=1)  #label predicts the image based on its probability

        label=np.where(LABELS[row][np.newaxis,:] == 1)[1][0] #theoretical label

        conf[label][Y_classes] += 1 #confusion matrix: C[i,j] increment the image number in class j knowing that it really belongs in class i

    for row in range (0,number_classes): #normalization : percentage of items i classified as class j
        for col in range (0,number_classes):
            if not np.sum(conf[row]) == 0: conf[row][col]=conf[row][col]/np.sum(conf[row]) #the confusion matrix is now normalized.
            #with row: how each class has been classified, i.e. C[i,j] / sum(conf[i]) of the class i objects are classified as class j
            #with column: what classes are  responsible for each classification, i.e. C[i,j] / sum(conf[j]) of the objects classified as class j were from class i

    return conf


def create_graph(conf):
    g = nx.Graph()  # graph creation
    above_max = np.where(conf > max_conf) #test against the threshold

    for row, col in zip(above_max[0],above_max[1]):
        if row != col:
            g.add_edge(row,col) #create graph
    return g



def from_graph_to_clusters(g,label_dict,nb_iter): #ajouter nb_iter
    sub_graphs = nx.connected_component_subgraphs(g) #sub_graphs contains the different connected structures

    clusters = [] #will contain the different clusters (numbers)
    clusters_with_names = [] ##will contain the different clusters (names)
    ID = []

    for index, sg in enumerate(sub_graphs):
        clusters.append(list(sg.nodes))

    list_number = -1
    for list_ in clusters:
        clusters_with_names.append([])
        list_number+=1
        for value in list_:
            name = ''
            for index in range(len(label_dict)):
                if (int(label_dict[index][-1])) == value:
                    name = label_dict[index][0]
            clusters_with_names[list_number].append(name)


    cluster_ID = -1
    list_size_clusters = []
    list_nodes_graph = []
    if clusters:

        path_iteration_dir = COMMON_PATH+'/clusters/iteration_'+str(nb_iter)

        if not os.path.isdir(path_iteration_dir):
            os.mkdir(path_iteration_dir)

        print("\n","creation of directories with the images sorted by cluster inside:")

        #list_nodes_graph = []
        for cluster in clusters_with_names:
            cluster_size=len(cluster)
            cluster_ID += 1
            ID.append(ID_static())

            path_cluster = COMMON_PATH+'/clusters/iteration_'+str(nb_iter)+'/cluster_n'+str(ID[cluster_ID])
            list_size_clusters.append(cluster_size)
            os.mkdir(path_cluster)

            cluster_created = 'cluster_n'+str(ID[cluster_ID])+'_s'+str(cluster_size)

            list_nodes_graph.append(cluster_created)

            print("fill cluster", cluster_ID, "...")

            for class_name in cluster:
                dst = COMMON_PATH+'/clusters/iteration_'+str(nb_iter)+'/cluster_n'+str(ID[cluster_ID])+'/'+class_name
                os.mkdir(dst)

                src = COMMON_PATH+'/clusters/iteration_0/cluster_n0/'+class_name

                distutils.dir_util.copy_tree(src, dst, preserve_mode=1)

        print("Success")
    return(cluster_ID,list_size_clusters,clusters,ID,list_nodes_graph)


def cluster_training(cluster_ID,list_size_clusters,nb_iter,clusters,ID):
    if clusters:

        print("Start Training")
        for cluster in range(0,cluster_ID+1):
            print("cluster number", cluster)
            # NUM_CLASSES
            NUM_CLASSES = list_size_clusters[cluster]

            # PATH_IMAGES
            PATH_IMAGES = COMMON_PATH+'/clusters/iteration_'+str(nb_iter)+'/cluster_n'+str(ID[cluster])
            print(PATH_IMAGES)

            # PATH_LABEL
            PATH_LABEL_ITER_DIR = COMMON_PATH+'/clusters_labels/iteration_'+str(nb_iter)
            if not os.path.isdir(PATH_LABEL_ITER_DIR):
                os.mkdir(PATH_LABEL_ITER_DIR)
            PATH_LABEL_DIR = COMMON_PATH+'/clusters_labels/iteration_'+str(nb_iter)+'/labels_cluster_'+str(ID[cluster])
            os.mkdir(PATH_LABEL_DIR)
            PATH_LABELS = PATH_LABEL_DIR+'/label'

            # PATH_SAVE_MODEL
            PATH_SAVE_ITER_DIR = COMMON_PATH+'/clusters_saves/iteration_'+str(nb_iter)
            if not os.path.isdir(PATH_SAVE_ITER_DIR):
                os.mkdir(PATH_SAVE_ITER_DIR)
            PATH_SAVE_MODEL_DIR = COMMON_PATH+'/clusters_saves/iteration_'+str(nb_iter)+'/save_cluster_'+str(ID[cluster])
            os.mkdir(PATH_SAVE_MODEL_DIR)
            PATH_SAVE_MODEL = PATH_SAVE_MODEL_DIR+'/save'

            # PREPARED_IMAGES_PATH
            PATH_PREPARED_IMAGES_ITER_DIR = COMMON_PATH+'/clusters_prepared_images/iteration_'+str(nb_iter)
            if not os.path.isdir(PATH_PREPARED_IMAGES_ITER_DIR):
                os.mkdir(PATH_PREPARED_IMAGES_ITER_DIR)
            PATH_PREPARED_IMAGES_DIR = COMMON_PATH+'/clusters_prepared_images/iteration_'+str(nb_iter)+'/prepared_images_cluster_'+str(ID[cluster])
            os.mkdir(PATH_PREPARED_IMAGES_DIR)
            PATH_PREPARED_IMAGES = PATH_PREPARED_IMAGES_DIR+'/prepared_images'

            # PATH TEST IMAGES
            PATH_TEST_IMAGES_ITER_DIR = COMMON_PATH+'/test_images/iteration_'+str(nb_iter)
            if not os.path.isdir(PATH_TEST_IMAGES_ITER_DIR):
                os.mkdir(PATH_TEST_IMAGES_ITER_DIR)
            PATH_TEST_IMAGES_DIR = COMMON_PATH+'/test_images/iteration_'+str(nb_iter)+'/test_images_cluster_'+str(ID[cluster])
            os.mkdir( PATH_TEST_IMAGES_DIR)
            PATH_TEST_IMAGES = PATH_TEST_IMAGES_DIR+'/test_images'

            # PATH TEST LABELS
            PATH_TEST_LABELS_ITER_DIR = COMMON_PATH+'/test_labels/iteration_'+str(nb_iter)
            if not os.path.isdir(PATH_TEST_LABELS_ITER_DIR):
                os.mkdir(PATH_TEST_LABELS_ITER_DIR)
            PATH_TEST_LABELS_DIR = COMMON_PATH+'/test_labels/iteration_'+str(nb_iter)+'/test_labels_cluster_'+str(ID[cluster])
            os.mkdir(PATH_TEST_LABELS_DIR)
            PATH_TEST_LABELS = PATH_TEST_LABELS_DIR+'/test_labels'

            # PATH LABEL DICT
            PATH_LABEL_DICT_ITER_DIR = COMMON_PATH+'/label_dict/iteration_'+str(nb_iter)
            if not os.path.isdir(PATH_LABEL_DICT_ITER_DIR):
                os.mkdir(PATH_LABEL_DICT_ITER_DIR)
            PATH_LABEL_DICT_DIR = COMMON_PATH+'/label_dict/iteration_'+str(nb_iter)+'/label_dict_cluster_'+str(ID[cluster])
            os.mkdir(PATH_LABEL_DICT_DIR)
            PATH_LABEL_DICT = PATH_LABEL_DICT_DIR+'/label_dict'


            architecture_a_plat(NUM_CLASSES, PATH_IMAGES, PATH_LABELS, PATH_SAVE_MODEL, PATH_PREPARED_IMAGES, PATH_TEST_IMAGES,PATH_TEST_LABELS, PATH_LABEL_DICT)

    else: print("no training needed regarding the threshold")




clusters = True
iteration = 0

while(clusters):

    iteration += 1
    clusters_for_an_iteration = COMMON_PATH+'/clusters/iteration_'+str(iteration-1)

    print("-------------------------------------------------------------------------------")
    print("--------------------------------- iteration", iteration - 1, "---------------------------------")
    print("-------------------------------------------------------------------------------")

    for cluster in list(os.listdir(clusters_for_an_iteration)):
        print("cluster :",cluster)

        nb_cluster = int((((cluster.split("/")[-1]).split("_")[1:3])[0]).split("n")[1])

        model = load_model(COMMON_PATH+'/clusters_saves/iteration_'+str(iteration-1)+'/save_cluster_'+str(nb_cluster)+'/save')

        label_dict = np.load(COMMON_PATH+'/label_dict/iteration_'+str(iteration-1)+'/label_dict_cluster_'+str(nb_cluster)+'/label_dict.npy')

        test_images = np.load(COMMON_PATH+'/test_images/iteration_'+str(iteration-1)+'/test_images_cluster_'+str(nb_cluster)+'/test_images.npy')

        test_labels = np.load(COMMON_PATH+'/test_labels/iteration_'+str(iteration-1)+'/test_labels_cluster_'+str(nb_cluster)+'/test_labels.npy')


        nb_classes = dict_information(label_dict)

        nb_img = get_number_images(test_images)

        conf_matrix = confusion_matrix(nb_classes, nb_img, model, test_images, test_labels)
        # plt.matshow(conf_matrix)
        # plt.show()

        G = create_graph(conf_matrix)
        # nx.draw_networkx(G)
        # plt.show()

        ID, list_size, clusters,unique_ID,list_nodes_graph = from_graph_to_clusters(G, label_dict, iteration)

        graph_archi.add_node(cluster,pos=(nb_cluster,iteration-1))

        for node in list_nodes_graph:
            graph_archi.add_node(node,pos=((node.split('_')[-2]).split('n')[-1],iteration))
            graph_archi.add_edge(node,cluster)


        cluster_training(ID, list_size, iteration, clusters, unique_ID)


print("training finished!")
# pos = nx.get_node_attributes(graph_archi,'pos')
# nx.draw_networkx(graph_archi,pos)
# plt.show()






