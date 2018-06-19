from keras.models import load_model
from keras.applications.vgg16 import decode_predictions
import numpy as np

model = load_model('/home/bpouthie/test_archi/plancton/save')

X = np.load('/home/bpouthie/test_archi/plancton/Kaggle.npy')

LABELS = np.load('/home/bpouthie/test_archi/plancton/Kaggle_labels.npy')

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


#print(LABELS[123][np.newaxis,:])

conf=np.zeros((121,121))


for line in range(0,30336):

    Y_prob = model.predict(X[line][np.newaxis, :]) #proba d'appartenance de l'image
    Y_classes = Y_prob.argmax(axis=1)  #label PREDIT de l'image via sa proba
    label=np.where(LABELS[line][np.newaxis,:] == 1)[1][0] #label THEORIQUE de l'image

    conf[label][Y_classes] += 1 #coeff C[i,j] incremente le nombre d'image en classe j sachant qu'elle appartient reelement en classe i

for line in range (0,122): #normalization : diviser chaque élément d'une ligne par la somme des élments de cette même ligne : pourcentage d'elements i classifiés comme étant de la classe j
    for col in range (0,122):
        conf[line][col]=conf[line][col]/np.sum(conf[line])



print(conf)


# a=np.array([[1.0, 2.0], [3.0, 4.0]])
# print(a)
#
# for line in range (0,2): #normalization
#      for col in range (0,2):
#         a[line][col]=(a[line][col])/np.sum(a[line])
#
# print(a)