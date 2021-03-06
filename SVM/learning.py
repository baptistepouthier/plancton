from parameters import *
import numpy as np
import networkx as nx
import os
import pandas as pd
from data import divide_images_and_labels
from SVM import *


graph_archi = nx.Graph()


def static_var(varname, value):
    def decorate(func):
        setattr(func, varname, value)
        return func
    return decorate


@static_var("counter", 0)
def ID_static():
    ID_static.counter += 1
    return(ID_static.counter)


def create_graph(conf,label_dict): #label_dict indicate the true label
    g = nx.Graph()  # graph creation
    above_max = np.where(conf > max_conf) #test against the threshold

    # print(label_dict)

    for row, col in zip(above_max[0],above_max[1]):
        if row != col:
            if bool(label_dict): g.add_edge(label_dict[row][0],label_dict[col][0]) #create graph
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


def cluster_training(clusters,ID,nb_iter,all_images,all_label_numbers,all_label_names,all_clusters_with_ID):

    list_nodes_graph = []

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
            cluster_names = []

            for index in cluster_index: #define the custer images and the labels

                cluster_images.append(all_images[index])
                cluster_labels.append(all_label_numbers[index])
                cluster_names.append(all_label_names[index])#

            print('cluster composed by', set(cluster_names))
            print('image number:',len(cluster_labels))





            label_dict = dict()


            # if len(cluster_labels) >= 4 * batch_size:

            #### transform labels in adapted one-hot encoding ####
            unique = list(set(cluster_labels))
            new_values = list(range(0, len(unique)))

            rectified_labels = []
            for element in cluster_labels:
                for value in unique:
                    if element == value: rectified_labels.append(new_values[unique.index(value)])
                    label_dict[new_values[unique.index(value)]] = [value,
                                                                       cluster_names[cluster_labels.index(value)]]


            if not label_dict in all_clusters_with_ID[3::4]:

                list_nodes_graph.append('save_cluster_'+str(ID[ID_index]))# +'_s' + str(NUM_CLASSES))

                X_train, Y_train, X_val, Y_val, X_test, Y_test = divide_images_and_labels(np.array(cluster_images,dtype='float32'), rectified_labels) #cluster_labels

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

                train_SVM(X_train, Y_train,PATH_SAVE_MODEL)

                print("DL model "+str(ID[ID_index])+" saved")
                print("cluster trained (DL)-\n")

            else : print(" /!\ cluster already present at the previous iteration, no training  /!\ ")


    else: print("no more cluster - no training needed")

    return(all_clusters_with_ID,list_nodes_graph)


