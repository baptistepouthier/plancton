from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


model = load_model('/home/bpouthie/test_archi/plancton/save')

X = np.load('/home/bpouthie/test_archi/plancton/Kaggle.npy')

LABELS = np.load('/home/bpouthie/test_archi/plancton/Kaggle_labels.npy')

print("processing...")


conf=np.zeros((121,121)) #confusion matrix

g=nx.Graph() #graph creation

max_conf = 0.5 #threshold


for row in range(0,30336):

    Y_prob = model.predict(X[row][np.newaxis, :]) #belonging probability of images
    Y_classes = Y_prob.argmax(axis=1)  #label predicts the image based on its probability
    label=np.where(LABELS[row][np.newaxis,:] == 1)[1][0] #theoretical label

    conf[label][Y_classes] += 1 #coeff C[i,j] increment the image number in class j knowing that it really belongs in class i

for row in range (0,121): #normalization : percentage of items i classified as class j
    for col in range (0,121):
        conf[row][col]=conf[row][col]/np.sum(conf[row])
        above_max = np.where(conf[row][col]> max_conf)

for row, col in zip(above_max[0],above_max[1]):
    if row != col:
        g.add_edge(row,col)


nx.draw_networkx(g)

plt.savefig('/home/bpouthie/test_archi/plancton/graph.png')



plt.matshow(conf)

plt.savefig('/home/bpouthie/test_archi/plancton/matrix.png')


# a=np.array([[1.0, 2.0], [3.0, 4.0]])
# print(a)
#
# for line in range (0,2): #normalization
#      for col in range (0,2):
#         a[line][col]=(a[line][col])/np.sum(a[line])
#
# print(a)