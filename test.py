from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import pyprind
import distutils.dir_util
from confusion import architecture_a_plat as cnn


COMMON_PATH = 'D:/Users/Baptiste Pouthier/Documents/partage'
max_conf = 0.5
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

    for row in pyprind.prog_percent(range(0,number_images)):

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

        path_iteration_dir = '%s/clusters/iteration_%s' % (COMMON_PATH,nb_iter)

        if not os.path.isdir(path_iteration_dir):
            os.mkdir(path_iteration_dir)

        print("\n","creation of directories with the images sorted by cluster inside:")

        #list_nodes_graph = []
        for cluster in clusters_with_names:
            cluster_size=len(cluster)
            cluster_ID += 1
            ID.append(ID_static())

            path_cluster = '%s/clusters/iteration_%s/cluster_n%s' % (COMMON_PATH,nb_iter, ID[cluster_ID])
            list_size_clusters.append(cluster_size)
            os.mkdir(path_cluster)

            cluster_created = 'cluster_n%s_s%s' % (ID[cluster_ID], cluster_size)

            list_nodes_graph.append(cluster_created)

            print("fill cluster", cluster_ID, "...")

            for class_name in cluster:
                dst = '%s/clusters/iteration_%s/cluster_n%s/%s' % (COMMON_PATH,nb_iter, ID[cluster_ID], class_name)
                os.mkdir(dst)
                #os.mkdir('D:/Users/Baptiste Pouthier/Documents/partage/clusters/iteration_%s/cluster_n%s_s%s/%s' % (nb_iter,cluster_ID,cluster_size,class_name))

                src = '%s/clusters/iteration_0/cluster_n0/%s' % (COMMON_PATH,class_name)

                #dst = 'D:/Users/Baptiste Pouthier/Documents/partage/clusters/iteration_%s/cluster_n%s_s%s/%s' % (nb_iter,cluster_ID,cluster_size,class_name)
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
            PATH_IMAGES = '%s/clusters/iteration_%s/cluster_n%s' % (COMMON_PATH, nb_iter, ID[cluster])
            #PATH_IMAGES = 'D:/Users/Baptiste Pouthier/Documents/partage/clusters/iteration_%s/cluster_n%s_s%s' % (nb_iter, cluster, list_size_clusters[cluster])
            print(PATH_IMAGES)

            # PATH_LABEL
            PATH_LABEL_ITER_DIR = '%s/clusters_labels/iteration_%s' % (COMMON_PATH, nb_iter)
            if not os.path.isdir(PATH_LABEL_ITER_DIR):
                os.mkdir(PATH_LABEL_ITER_DIR)
            PATH_LABEL_DIR = '%s/clusters_labels/iteration_%s/labels_cluster_%s' % (COMMON_PATH, iteration, ID[cluster])
            os.mkdir(PATH_LABEL_DIR)
            PATH_LABELS = '%s/label' % (PATH_LABEL_DIR)

            # PATH_SAVE_MODEL
            PATH_SAVE_ITER_DIR = '%s/clusters_saves/iteration_%s' % (COMMON_PATH, nb_iter)
            if not os.path.isdir(PATH_SAVE_ITER_DIR):
                os.mkdir(PATH_SAVE_ITER_DIR)
            PATH_SAVE_MODEL_DIR = '%s/clusters_saves/iteration_%s/save_cluster_%s' % (COMMON_PATH,iteration, ID[cluster])
            os.mkdir(PATH_SAVE_MODEL_DIR)
            PATH_SAVE_MODEL = '%s/save' % (PATH_SAVE_MODEL_DIR)

            # PREPARED_IMAGES_PATH
            PATH_PREPARED_IMAGES_ITER_DIR = '%s/clusters_prepared_images/iteration_%s' % (COMMON_PATH,nb_iter)
            if not os.path.isdir(PATH_PREPARED_IMAGES_ITER_DIR):
                os.mkdir(PATH_PREPARED_IMAGES_ITER_DIR)
            PATH_PREPARED_IMAGES_DIR = '%s/clusters_prepared_images/iteration_%s/prepared_images_cluster_%s' % (COMMON_PATH,iteration, ID[cluster])
            os.mkdir(PATH_PREPARED_IMAGES_DIR)
            PATH_PREPARED_IMAGES = '%s/prepared_images' % (PATH_PREPARED_IMAGES_DIR)

            # PATH TEST IMAGES
            PATH_TEST_IMAGES_ITER_DIR = '%s/test_images/iteration_%s' % (COMMON_PATH, nb_iter)
            if not os.path.isdir(PATH_TEST_IMAGES_ITER_DIR):
                os.mkdir(PATH_TEST_IMAGES_ITER_DIR)
            PATH_TEST_IMAGES_DIR = '%s/test_images/iteration_%s/test_images_cluster_%s' % (COMMON_PATH, iteration, ID[cluster])
            os.mkdir( PATH_TEST_IMAGES_DIR)
            PATH_TEST_IMAGES = '%s/test_images' % (PATH_TEST_IMAGES_DIR)

            # PATH TEST LABELS
            PATH_TEST_LABELS_ITER_DIR = '%s/test_labels/iteration_%s' % (COMMON_PATH, nb_iter)
            if not os.path.isdir(PATH_TEST_LABELS_ITER_DIR):
                os.mkdir(PATH_TEST_LABELS_ITER_DIR)
            PATH_TEST_LABELS_DIR = '%s/test_labels/iteration_%s/test_labels_cluster_%s' % (COMMON_PATH, iteration, ID[cluster])
            os.mkdir(PATH_TEST_LABELS_DIR)
            PATH_TEST_LABELS = '%s/test_labels' % (PATH_TEST_LABELS_DIR)

            # PATH LABEL DICT
            PATH_LABEL_DICT_ITER_DIR = '%s/label_dict/iteration_%s' % (COMMON_PATH, nb_iter)
            if not os.path.isdir(PATH_LABEL_DICT_ITER_DIR):
                os.mkdir(PATH_LABEL_DICT_ITER_DIR)
            PATH_LABEL_DICT_DIR = '%s/label_dict/iteration_%s/label_dict_cluster_%s' % (COMMON_PATH, iteration, ID[cluster])
            os.mkdir(PATH_LABEL_DICT_DIR)
            PATH_LABEL_DICT = '%s/label_dict' % (PATH_LABEL_DICT_DIR)


            cnn.architecture_a_plat(NUM_CLASSES, PATH_IMAGES, PATH_LABELS, PATH_SAVE_MODEL, PATH_PREPARED_IMAGES, PATH_TEST_IMAGES,PATH_TEST_LABELS, PATH_LABEL_DICT)

    else: print("no training needed regarding the threshold")




clusters = True
iteration = 0

while(clusters):

    iteration += 1
    clusters_for_an_iteration = '%s/clusters/iteration_%s' % (COMMON_PATH, iteration-1)

    print("-------------------------------------------------------------------------------")
    print("--------------------------------- iteration", iteration - 1, "---------------------------------")
    print("-------------------------------------------------------------------------------")

    for cluster in list(os.listdir(clusters_for_an_iteration)):
        print("cluster :",cluster)

        nb_cluster = int((((cluster.split("/")[-1]).split("_")[1:3])[0]).split("n")[1])

        model = load_model('%s/clusters_saves/iteration_%s/save_cluster_%s/save' % (COMMON_PATH, iteration-1,nb_cluster))

        label_dict = np.load('%s/label_dict/iteration_%s/label_dict_cluster_%s/label_dict.npy'  % (COMMON_PATH ,iteration-1,nb_cluster))

        test_images = np.load('%s/test_images/iteration_%s/test_images_cluster_%s/test_images.npy'  % (COMMON_PATH, iteration-1,nb_cluster))

        test_labels = np.load('%s/test_labels/iteration_%s/test_labels_cluster_%s/test_labels.npy' % (COMMON_PATH, iteration - 1, nb_cluster))


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






