from keras.models import load_model
from keras.applications.vgg16 import decode_predictions
import numpy as np

model = load_model("/home/bpouthie/test_archi/plancton/save")

X = np.load("/home/bpouthie/test_archi/plancton/Kaggle.npy");

print("processing...")
Y = model.predict_classes(X)

print(Y)
#print('Top 3 :', decode_predictions(Y, top=3)[0])
