import numpy as np
from parameters import *
from learning import create_graph, from_graph_to_clusters, cluster_training
from analysis import conf_matrix
from prediction import multiple_predictions
import os
from keras.models import load_model
from CNN import cnn
from data import prepare_images, divide_images_and_labels
import networkx as nx
import matplotlib.pyplot as plt




###############################################################################

if run_from_zero:
    all_images, all_label_names, all_label_numbers = prepare_images()

    # save
    np.save(COMMON_PATH + "/all_images", all_images)
    np.save(COMMON_PATH + "/all_label_names", all_label_names)
    np.save(COMMON_PATH + "/all_label_numbers", all_label_numbers)

    X_train, Y_train, X_val, Y_val, X_test, Y_test = divide_images_and_labels(np.array(all_images, dtype='float32'),np.eye(121, dtype='uint8')[all_label_numbers])

    # save
    np.save(COMMON_PATH + "/X_test", X_test)
    np.save(COMMON_PATH + "/Y_test", Y_test)

    cnn(121, X_train, Y_train, X_test, Y_test, X_val, Y_val, COMMON_PATH+'/clusters_saves/iteration_0/save_cluster_0/save',batch_size,img_size,None)

else:
    all_images = np.load(COMMON_PATH + '/all_images.npy')
    all_label_names = np.load(COMMON_PATH + '/all_label_names.npy')
    all_label_numbers = np.load(COMMON_PATH + '/all_label_numbers.npy')

    X_test = np.load(COMMON_PATH+'/X_test.npy')
    Y_test = np.load(COMMON_PATH+'/Y_test.npy')


empty_dict=dict()
all_clusters_with_ID=[0,X_test,Y_test,empty_dict]
clusters = True

graph_archi = nx.Graph()

iteration = 0
nb_cluster = 0
# label_dict_graph={}
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

            if not isinstance(all_clusters_with_ID[all_clusters_with_ID[::4].index(nb_cluster) * 4 + 1], int):

                X_test, Y_test = all_clusters_with_ID[all_clusters_with_ID[::4].index(nb_cluster)*4+1],all_clusters_with_ID[all_clusters_with_ID[::4].index(nb_cluster)*4+2]

                dict_references=all_clusters_with_ID[all_clusters_with_ID[::4].index(nb_cluster)*4+3]


                model = load_model(COMMON_PATH+'/clusters_saves/iteration_'+str(iteration-1)+'/'+cluster+'/save')
                model_path = COMMON_PATH+'/clusters_saves/iteration_'+str(iteration-1)+'/'+cluster+'/save'

                confus_matrix = conf_matrix(model, X_test, Y_test)


                G = create_graph(confus_matrix,dict_references)
                # nx.draw_networkx(G)
                # plt.show()

                clusters, ID = from_graph_to_clusters(G)

                all_clusters_with_ID, list_nodes_graph = cluster_training(clusters,ID,iteration,all_images,all_label_numbers,all_label_names,all_clusters_with_ID,model_path)

                graph_archi.add_node(cluster)
                # label_dict_graph[cluster] = nb_cluster
                for node in list_nodes_graph:
                    graph_archi.add_node(node)
                    # label_dict_graph[node] = int(node.split('_')[-1])
                    graph_archi.add_edge(cluster, node)


            else :

                print("cluster trained with SVM")

                graph_archi.add_node(cluster)
                # label_dict_graph[cluster] = nb_cluster
                graph_archi.add_edge(cluster,'save_cluster_'+str(all_clusters_with_ID[all_clusters_with_ID[::4].index(nb_cluster) * 4 + 1]))


    else :
        print("\nThere are only redundant groups")
        clusters = False



print("training finished!\n")

if show_network:

    plt.title('Architecture graph of the classifier')
    plt.axis('off')
    labeldict = {}
    for iter in os.listdir(COMMON_PATH+'/clusters_saves')[1::]:
        clusters = os.listdir(COMMON_PATH+'/clusters_saves/' + iter)
        for cluster in clusters:
            labeldict[str(cluster)]='\n'.join(np.genfromtxt(COMMON_PATH+'/clusters_saves/'+ iter +'/'+ cluster +'/'+ 'labels.csv', dtype=None, encoding=None)[1:])

    nx.draw_networkx(graph_archi,labels=labeldict)
    plt.show()


X_test = np.load(COMMON_PATH+'/X_test.npy')
Y_test = np.load(COMMON_PATH+'/Y_test.npy')
model = load_model(COMMON_PATH+'/clusters_saves/iteration_0/save_cluster_0/save')

acc_hierachical = multiple_predictions(X_test, Y_test,all_label_names,all_label_numbers,COMMON_PATH,all_clusters_with_ID)
print("accuracy with hierarchy:",acc_hierachical)


Y_pred = model.predict(X_test)
predClasses = Y_pred.argmax(axis=-1)
trueClasses = Y_test.argmax(axis=-1)

acc = np.sum(predClasses == trueClasses) / np.size(predClasses)
print("accuracy with 'flat' cnn:",acc)






