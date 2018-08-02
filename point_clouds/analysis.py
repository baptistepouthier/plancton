from parameters import *
from sklearn.metrics import confusion_matrix
import sklearn.preprocessing
import matplotlib.pyplot as plt


def conf_matrix(model,images,labels):

    print("predictions by the model with",len(images),"images")
    prediction = model.predict(images)


    conf = confusion_matrix(labels,prediction)

    conf = sklearn.preprocessing.normalize(conf, axis=1, norm='l1')

    if show_confusion_matrix:
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
