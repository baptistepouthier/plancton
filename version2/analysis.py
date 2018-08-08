from sklearn.metrics import confusion_matrix
import sklearn.preprocessing
import numpy as np
import matplotlib.pyplot as plt
from parameters import *

def conf_matrix(model,images,labels):

    print("predictions by the model with",len(images),"images")
    Y_prob = model.predict(images)  # belonging probability of images
    Y_classes = Y_prob.argmax(axis=1)  # label predicts the image based on its probability

    true_label = []
    for row in range(0, len(images)):
        true_label.append(np.where(labels[row][np.newaxis, :] == 1)[1][0])  # theoretical label

    conf = confusion_matrix(true_label,Y_classes)

    conf = sklearn.preprocessing.normalize(conf, axis=1, norm='l1')

    if show_conf_matrix:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(conf,cmap='plasma')
        plt.title('Confusion matrix of the classifier')
        fig.colorbar(cax)
        plt.xlabel('Predicted')
        plt.ylabel('True')

        plt.show()
    #the confusion matrix is now normalized.
    #with row: how each class has been classified, i.e. C[i,j] / sum(conf[i]) of the class i objects are classified as class j
    #with column: what classes are  responsible for each classification, i.e. C[i,j] / sum(conf[j]) of the objects classified as class j were from class i

    return conf
