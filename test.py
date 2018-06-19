from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

model = load_model("C:/Users/Baptiste Pouthier/Documents/partage/Architecture/save")

X = np.load("C:/Users/Baptiste Pouthier/Documents/partage/Architecture/Kaggle.npy")

LABELS = np.load("C:/Users/Baptiste Pouthier/Documents/partage/Architecture/Kaggle_labels.npy")



print("processing...")
#Y_prob = model.predict(X[123][np.newaxis,:])
#Y_prob=model.predict(X)
#Y_classes = Y_prob.argmax(axis=1) #label de l'image

#print("represent probabilities at each categories for each row (one row = one image) :")
#print(Y_prob)

#print("shape of the matrix :")
#print(Y_prob.size)

#print("label : ")
#print(Y_classes)


#print(LABELS[123][np.axis,:])

conf=np.zeros((121,121)) #confusion matrix


for row in range(0,30336):

    Y_prob = model.predict(X[row][np.newaxis, :]) #proba d'appartenance de l'image
    Y_classes = Y_prob.argmax(axis=1)  #label PREDIT de l'image via sa proba
    label=np.where(LABELS[row][np.newaxis,:] == 1)[1][0] #label THEORIQUE de l'image

    conf[label][Y_classes] += 1 #coeff C[i,j] incremente le nombre d'image en classe j sachant qu'elle appartient reelement en classe i

for row in range (0,121): #normalization : diviser chaque élément d'une ligne par la somme des élments de cette même ligne : pourcentage d'elements i classifiés comme étant de la classe j
    for col in range (0,121):
        conf[row][col]=conf[row][col]/np.sum(conf[row])



#print(conf)
plt.matshow(conf)

plt.show()


#TEST-TEST-TEST-TEST-TEST-TEST-TEST-TEST
# a=np.array([[1.0, 2.0], [3.0, 4.0]])
# print(a)
#
# for row in range (0,2): #normalization
#      for col in range (0,2):
#         a[row][col]=(a[row][col])/np.sum(a[row])
#
# print(a)