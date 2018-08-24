from learning import create_graph, from_graph_to_clusters, cluster_training
from analysis import conf_matrix
from prediction import multiple_predictions
import os
import glob
from SVM import *
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from prepare_data_sift import get_class, gen_sift_features, train_test_val_split_idxs, cluster_features
import networkx as nx
from data import plot_dataset, create_dataset, divide_images_and_labels
from parameters import *
import numpy as np
import matplotlib.pyplot as plt

if synthetic_data :
    dataWithLabel, input_data, all_label_numbers = create_dataset()
    all_label_names = all_label_numbers
    plot_dataset(input_data,dataWithLabel)

else :

    if from_zero :
        all_img_path = glob.glob(os.path.join(root_dir, '*' + os.sep + '*.jpg'))

        labeled_image_paths = []

        for path in all_img_path: labeled_image_paths.append([path, get_class(path)])

        img_descs,  all_label_names, all_label_numbers = gen_sift_features(labeled_image_paths)

        total_rows = len(img_descs)
        training_idxs, test_idxs, val_idxs = train_test_val_split_idxs(total_rows, 0,0)  # first number: test, second: validation, rest: training

        input_data, cluster_model = cluster_features(img_descs, training_idxs, KMeans(n_clusters=500))

        print('save sift features...')
        np.save(COMMON_PATH+'/test_sift/input_data',input_data)
        np.save(COMMON_PATH +'/test_sift/all_label_numbers',all_label_numbers)
        np.save(COMMON_PATH +'/test_sift/all_label_names',all_label_names)

    else :

        print('loading data...')
        input_data = np.load(COMMON_PATH+'/test_sift/input_data.npy')
        all_label_numbers = np.load(COMMON_PATH +'/test_sift/all_label_numbers.npy')
        all_label_names = np.load(COMMON_PATH +'/test_sift/all_label_names.npy')


###############################################################################


X_train, Y_train, X_val, Y_val, X_test, Y_test = divide_images_and_labels(input_data, all_label_numbers)

X_test2 = X_test
Y_test2 = Y_test

train_SVM(X_train, Y_train,COMMON_PATH+'/clusters_saves/iteration_0/save_cluster_0/save')

graph_archi = nx.Graph()

empty_dict=dict()
all_clusters_with_ID=[0,X_test,Y_test,empty_dict]
clusters = True
iteration = 0
nb_cluster = 0
label_dict_graph={}
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

                model = joblib.load(COMMON_PATH+'/clusters_saves/iteration_'+str(iteration-1)+'/'+cluster+'/save')


                confus_matrix = conf_matrix(model, X_test, Y_test)


                G = create_graph(confus_matrix,dict_references)
                # nx.draw_networkx(G)
                # plt.show()

                clusters, ID = from_graph_to_clusters(G)



                all_clusters_with_ID,list_nodes_graph = cluster_training(clusters,ID,iteration,input_data,all_label_numbers,all_label_names,all_clusters_with_ID)

                graph_archi.add_node(cluster, pos=(nb_cluster, iteration - 1))
                label_dict_graph[cluster]=nb_cluster
                for node in list_nodes_graph:
                    graph_archi.add_node(node, pos=((node.split('_')[-1]), iteration))
                    label_dict_graph[node] =  int(node.split('_')[-1])
                    graph_archi.add_edge(cluster,node)

            else : print("cluster trained with SVM")

    else :
        print("\nThere are only redundant groups")
        clusters = False

#np.save(COMMON_PATH+'/clust_with_ID',all_clusters_with_ID)


print("training finished!\n")

if show_network:
    pos = nx.get_node_attributes(graph_archi,'pos')
    plt.yticks(np.arange(0, iteration, dtype=np.int))
    plt.xticks(np.arange(0, nb_cluster+1, dtype=np.int))
    plt.xlabel('cluster number')
    plt.ylabel('iteration')
    plt.title('Architecture graph of the classifier')
    nx.draw_networkx(graph_archi,pos,labels=label_dict_graph)
    plt.show()

acc = multiple_predictions(X_test2, Y_test2,all_label_names,all_label_numbers,COMMON_PATH,all_clusters_with_ID)
print("final acc:",acc)

##################################################################################################"


# pos = nx.get_node_attributes(graph_archi,'pos')
# nx.draw_networkx(graph_archi,pos)
# plt.show()





