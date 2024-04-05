from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict


def naive_bayes_classification(matrix, classes):
    nb_classifier = MultinomialNB()
    y_pred = cross_val_predict(nb_classifier, matrix, classes, cv=10)
    return y_pred


def svm_classification(matrix, classes):
    svm_classifier = SVC(kernel='linear')
    y_pred = cross_val_predict(svm_classifier, matrix, classes, cv=10)
    return y_pred

def random_forest_classification(matrix, classes):
    rf_classifier = RandomForestClassifier()
    y_pred = cross_val_predict(rf_classifier, matrix, classes, cv=10)
    return y_pred
