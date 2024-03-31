#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 11:27:53 2024

@author: pauline
"""

import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from linear_regression import linear_regression


class FileLoader:
    def load(self, path):
        """transforme le csv en dataframe"""

        df = pd.read_csv(path, sep="|")
        rows, columns = df.shape
        print(f"Loading dataset of dimensions {rows} x {columns}")

        return df


def get_data(df):
    """Extrait les phrases et leurs classes depuis un dataframe"""

    data = df[["Texte", "Score"]]
    phrases = data["Texte"]
    y = data["Score"]

    return phrases, y


def get_matrix(phrases):
    """vectorise les phrases"""

    vectorizer = TfidfVectorizer(input="content")
    matrix = vectorizer.fit_transform(phrases).toarray()

    return matrix


def print_error():
    print("Choississez un classifieur : option -c suivi du modele choisi :")
    print("-c LR \tpour lancer Linear Regression")


# Dictionnaire pour lancer le classifieur en fonction de l'option choisie
modeles = {
    "LR": linear_regression,
    "SVM": "plop",
    "DT": "plop",
    "RF": "plop",
}

def save_report(report, output):
    with open(output, "w") as file:
        file.write(report)

def regression_report(y, y_hat, threshold=0.5):
    mse = mean_squared_error(y, y_hat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_hat)
    r2 = r2_score(y, y_hat)
    accuracy = create_accuracy_score(y, y_hat, threshold)
    report = {
        "Mean Squared Error:": mse,
        "Root MSE:": rmse,
        "Mean Absolute Error:": mae,
        "R squared:": r2,
        f"Pseudo-accuracy ({threshold}):": accuracy
    }
    final_report = ""
    for metric, value in report.items():
        final_report += f"{metric.ljust(30)}{value}\n"
    return final_report


def create_accuracy_score(y, y_hat, threshold):
    correct_predictions = sum(abs(y - y_hat) <= threshold)
    accuracy = (correct_predictions / len(y)) * 100
    return accuracy


################################ MAIN ####################################


def main():
    parser = argparse.ArgumentParser(description="lanceur de classifieurs")
    parser.add_argument(
        "input_file", type=str, help="Chemin vers le fichier contenant les data csv"
    )
    parser.add_argument(
        "-c",
        "--classifieur",
        choices=["LR"],
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
    phrases, y = get_data(file)
    matrix = get_matrix(phrases)

    # Lancement du classifieur
    y_hat = modeles[args.classifieur](matrix, y)
    report = regression_report(y, y_hat)
    save_report(report, output)


if __name__ == "__main__":
    main()
