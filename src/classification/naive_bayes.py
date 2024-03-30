import sys
from pretaitement import get_data, FileLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import numpy as np


def get_matrix(corpus):
    vectorizer = TfidfVectorizer(input="content", stop_words="english")
    matrix = vectorizer.fit_transform(corpus).toarray()
    return matrix

def naive_bayes_classification(matrix, classes):
    nb_classifier = MultinomialNB()
    y_pred = cross_val_predict(nb_classifier, matrix, classes, cv=10)
    accuracy = accuracy_score(classes, y_pred)
    return y_pred


def main():
    if len(sys.argv) != 2:
        sys.exit("Il faut un csv en argument.")
    loader = FileLoader()
    file = loader.load(sys.argv[1])
    corpus, classes = get_data(file)
    matrix = get_matrix(corpus)
    y_pred = naive_bayes_classification(matrix, classes)

if __name__ == "__main__":
    main()
