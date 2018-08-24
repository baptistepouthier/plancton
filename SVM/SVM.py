from sklearn.externals import joblib
from sklearn import svm


def train_SVM(X_train,Y_train,path_save):
    clf = svm.LinearSVC()
    clf.fit(X_train, Y_train)
    joblib.dump(clf, path_save)


