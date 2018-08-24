from parameters import *
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

data = []
nb_pts = 200
varx=0.25
vary=0.25


def divide_images_and_labels(images, labels):

    X_train, X_val, Y_train, Y_val = train_test_split(images, labels, test_size=0.2) #images :all_images     labels: all_label_numbers
    X_val, X_test, Y_val, Y_test = train_test_split(X_val, Y_val, test_size=0.5)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def create_dataset():


    cov = [[varx, 0,0],[0, vary,0],[0,0,0]]

    mean = [[0,7,0],[5,5.5,1],[2.5,6,2],[1.3,7.5,3],[2.7,7.2,4],[0.8,5.75,5],[4,6.5,6],[2,5,7],[4.2,7.6,8],[3.5,5.1,9]]

    for i in range(len(mean)):
        # dataset['data'+str(i)] = np.random.multivariate_normal(mean[i], cov, N)
        data.append(np.random.multivariate_normal(mean[i], cov, nb_pts))

    dataWithLabel=np.random.permutation(np.concatenate(data))

    #D : données
    all_points = dataWithLabel[:,[0,1]]



    #Z : Labels des données
    all_label_numbers = dataWithLabel[:,2]



    return dataWithLabel, all_points, all_label_numbers



def plot_dataset(all_images,dataWithLabel):
    list_colors = ['red', 'green', 'blue', 'orange', 'black', 'purple', 'yellow', '#bd2309', 'cyan', 'm']
    for i in range(len(dataWithLabel)):
        for index in range(len(list_colors)):
            if dataWithLabel[i,2]==index :
                plt.scatter(all_images[i, 0], all_images[i, 1], c=list_colors[index], marker='x')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('samples')
    plt.show()


