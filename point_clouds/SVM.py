from sklearn.externals import joblib
from sklearn import svm
from data import *

# dataWithLabel, all_points, all_label_numbers = create_dataset()
#
# X_train, Y_train, X_val, Y_val, X_test, Y_test = divide_images_and_labels(np.array(all_points), all_label_numbers)

def train_SVM(X_train,Y_train,path_save):
    clf = svm.LinearSVC()
    clf.fit(X_train, Y_train)
    joblib.dump(clf, path_save)
    #return clf


# clf = train_SVM(X_train,Y_train,'path_save')
# #
# print(clf.predict(X_test[0]))
#
# true_label = []
# for row in range(0, len(Y_test)):
#     true_label.append(Y_test[row])  # theoretical label
#
# print(true_label)