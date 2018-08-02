import numpy as np
import os
from keras.models import load_model
from sklearn.externals import joblib


def prediction(data_test,label_test,model_0,list_dir_iteration,all_label_names,all_label_numbers,COMMON_PATH,all_clusters_with_ID_reduced,row):

    Y_classes = model_0.predict(data_test)[row]
    label_name = all_label_names[list(all_label_numbers).index(Y_classes)]

    for iteration in list_dir_iteration[1::]:  # for each iteration directory

        species_found = False

        list_clusters = sorted(os.listdir(COMMON_PATH + '/clusters_saves' + '/' + iteration),
                               key=lambda k: int(k.split("_")[-1]))

        for cluster in list_clusters:  # for each cluster in the iteration directory


            if not species_found:
                cluster_species = np.genfromtxt(
                    COMMON_PATH + '/clusters_saves' + '/' + iteration + '/' + cluster + '/labels.csv', dtype=None,
                    encoding=None)[1:]


                if label_name in cluster_species:
                    # print(cluster_species)
                    species_found = True
                    ID = int(cluster.split('_')[-1])

                    model = joblib.load(COMMON_PATH + '/clusters_saves' + '/' + iteration + '/' + cluster + '/save')
                    prediction = model.predict(data_test)[row]

                    label_name = all_clusters_with_ID_reduced[all_clusters_with_ID_reduced.index(ID) + 1].get(int(prediction))[-1]  # image's label name


    true_label = label_test[row]  # theoretical label
    true_label_name = all_label_names[list(all_label_numbers).index(true_label)]

    return label_name, true_label_name

def multiple_predictions(data_test, label_test,all_label_names,all_label_numbers,COMMON_PATH,all_clusters_with_ID):

    all_clusters_with_ID_reduced = [x for x in all_clusters_with_ID if isinstance(x, int) or isinstance(x, dict)]

    print("final accuracy :")
    well_sorted = 0
    model_0 = joblib.load(COMMON_PATH+'/clusters_saves/iteration_0/save_cluster_0/save')  # model 0
    cluster_save_dir = COMMON_PATH+'/clusters_saves'
    list_dir_iteration = sorted(os.listdir(cluster_save_dir), key=lambda k: int(k.split("_")[-1]))
    number_images = len(data_test)

    for row in range(0, number_images):  # for each image
        if row % 10 == 0 : print("image",row,'/',number_images)

        label_name, true_label_name = prediction(data_test,label_test, model_0, list_dir_iteration,all_label_names,all_label_numbers,COMMON_PATH,all_clusters_with_ID_reduced,row)


        if true_label_name == label_name: well_sorted += 1
        if row % 10 == 0 and row != 0 : print(well_sorted / row)  # progress

    acc = well_sorted / number_images
    return acc