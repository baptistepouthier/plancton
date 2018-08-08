from keras.models import load_model
import numpy as np
import pyprind
import os
import glob


root_path = 'D:/Users/Baptiste Pouthier/Documents/partage'

data_test = np.load('%s/test_images/iteration_0/test_images_cluster_0/test_images.npy' % root_path)
label_test = np.load('%s/test_labels/iteration_0/test_labels_cluster_0/test_labels.npy' % root_path)
model_0 = load_model('%s/clusters_saves/iteration_0/save_cluster_0/save' % root_path) #model 0
number_images = len(data_test)


cluster_save_dir = '%s/clusters_saves' % root_path






def prediction(model_0,data_test,label_test):

    well_sorted = 0
    model_0 = load_model('%s/clusters_saves/iteration_0/save_cluster_0/save' % root_path)  # model 0

    list_dir_iteration = sorted(os.listdir(cluster_save_dir), key=lambda k: int(k.split("_")[-1]))
    number_images = len(data_test)
    for row in pyprind.prog_percent(range(0, number_images)): #for each image

        Y_prob = model_0.predict(data_test[row][np.newaxis, :])
        Y_classes = Y_prob.argmax(axis=1)  # predicts the image label based on the best probability
        label_name = all_label_names[all_label_numbers.index(Y_classes)]


        for iteration in list_dir_iteration: #for each iteration directory

            species_found = False

            list_clusters = sorted(os.listdir(root_path+'/clusters_saves'+'/'+iteration), key=lambda k: int(k.split("_")[-1]))

            for cluster in list_clusters: #for each cluster in the iteration directory

                if not species_found:
                    cluster_species = np.genfromtxt(root_path+'/clusters_saves'+'/'+iteration+'/'+cluster+'/labels.csv', dtype=None, encoding=None)[1:]
                    if label_name in cluster_species :
                        species_found = True
                        model = load_model(root_path+'/clusters_saves'+'/'+iteration+'/'+cluster+'save')
                        Y_prob = model.predict(data_test[row][np.newaxis, :])
                        Y_classes = Y_prob.argmax(axis=1)  # predicts the image label based on the best probability
                        ID = int(cluster.split('_')[-1])
                        label_name = all_index_with_ID[all_index_with_ID.index(ID) + 1][Y_classes][1] #image's label name


        true_label = np.where(label_test[row][np.newaxis, :] == 1)[1][0]  # theoretical label
        true_label_name = all_label_names[all_label_numbers.index(true_label)]

        if true_label_name == label_name: well_sorted += 1
        if not row == 0: print(well_sorted / row)  # progress


    acc = well_sorted/number_images
    return acc




