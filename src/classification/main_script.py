#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 11:27:53 2024

@author: pauline
"""

import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from algorithms import naive_bayes_classification, svm_classification
# from naive_bayes import naive_bayes_classification
# from svm import svm_classification


class FileLoader:
    def load(self, path):
        """transforme le csv en dataframe"""

        df = pd.read_csv(path, sep="|")
        rows, columns = df.shape
        print(f"Loading dataset of dimensions {rows} x {columns}")

        return df


def get_data(df):
    """Extrait les phrases et leurs classes depuis un dataframe"""

    data = df[["Texte", "Classe"]]
    phrases = data["Texte"]
    classes = data["Classe"]

    return phrases, classes


def get_matrix(phrases):
    """vectorise les phrases"""

    vectorizer = TfidfVectorizer(input="content")
    matrix = vectorizer.fit_transform(phrases).toarray()

    return matrix


def print_error():
    print("Choississez un classifieur : option -c suivi du modele choisi :")
    print("-c NB \tpour lancer Naive Bayes")
    print("-c SVM \tpour lancer SVM")
    print("-c DT \tpour lancer Decision Tree")
    print("-c RF \tpour lancer Random Forest")


# Dictionnaire pour lancer le classifieur en fonction de l'option choisie
modeles = {
    "NB": naive_bayes_classification,
    "SVM": svm_classification,
    "DT": "plop",
    "RF": "plop",
}


def save_prediction(prediction, classes, output):
    report = str(classification_report(classes, prediction))
    with open(output, "w") as file:
        file.write(report)


def generate_confusion_matrix(prediction, classes):
    conf_matrix = confusion_matrix(classes, prediction)
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=np.unique(classes))
    disp.plot(cmap="OrRd")
    plt.show()


################################ MAIN ####################################


def main():
    parser = argparse.ArgumentParser(description="lanceur de classifieurs")
    parser.add_argument(
        "input_file", type=str, help="Chemin vers le fichier contenant les data csv"
    )
    parser.add_argument(
        "-c",
        "--classifieur",
        choices=["NB", "SVM", "DT", "RF"],
        help="Choix du modèle de classifieur",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="Chemin vers le fichier pour sauvegarder les scores.",
    )
    args = parser.parse_args()
    if not args.classifieur:
        print_error()
        return 1
    output = args.output_file
    if not output:
        output = f"classification-report-{args.classifieur}.txt"
    # Load le corpus et extrait les vecteurs (matrices) et les classes
    loader = FileLoader()
    file = loader.load(args.input_file)
    phrases, classes = get_data(file)
    matrix = get_matrix(phrases)

    # Lancement du classifieur
    prediction = modeles[args.classifieur](matrix, classes)
    save_prediction(prediction, classes, output)
    generate_confusion_matrix(prediction, classes)


if __name__ == "__main__":
    main()
