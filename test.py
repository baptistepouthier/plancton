from keras.models import load_model
from keras.applications.vgg16 import decode_predictions
import numpy as np

model = load_model("C:/Users/Baptiste Pouthier/Documents/partage/Architecture/save")

X = np.load("C:/Users/Baptiste Pouthier/Documents/partage/Architecture/Kaggle.npy");

print("processing...")
Y = model.predict_classes(X)

print(Y)
#print('Top 3 :', decode_predictions(Y, top=3)[0])
