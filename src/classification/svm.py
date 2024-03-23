import sys
from pretaitement import get_data, FileLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np


def get_matrix(corpus):
    vectorizer = TfidfVectorizer(input="content", stop_words="english")
    matrix = vectorizer.fit_transform(corpus).toarray()
    return matrix

def svm_classification(matrix, classes):
    svm_classifier = SVC(kernel='linear')
    y_pred = cross_val_predict(svm_classifier, matrix, classes, cv=10)
    accuracy = accuracy_score(classes, y_pred)
    precision = precision_score(classes, y_pred, average="macro")
    recall = recall_score(classes, y_pred, average="macro")
    f_score = f1_score(classes, y_pred, average="macro")
    conf_matrix = confusion_matrix(classes, y_pred)
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=np.unique(classes))
    disp.plot(cmap="OrRd")
    plt.show()
    return accuracy, precision, recall, f_score

def main():
    if len(sys.argv) != 2:
        sys.exit("Il faut un csv en argument.")
    loader = FileLoader()
    file = loader.load(sys.argv[1])
    corpus, classes = get_data(file)
    matrix = get_matrix(corpus)
    accuracy, precision, recall, f_score = svm_classification(matrix, classes)
    print(f"Accuracy: {accuracy}")
    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"f_score: {f_score}")

if __name__ == "__main__":
    main()
