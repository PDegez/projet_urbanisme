import sys
from pretaitement import get_data, FileLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def get_matrix(corpus):
    vectorizer = TfidfVectorizer(input="content", stop_words="english")
    matrix = vectorizer.fit_transform(corpus).toarray()
    return matrix


def calculate_PCA(matrix):
    pca = PCA()
    pca_matrix = pca.fit_transform(matrix)
    exp_var_ratio = pca.explained_variance_ratio_
    cum_sum_eigenvalues = np.cumsum(exp_var_ratio)
    return cum_sum_eigenvalues


def plot_var_ratio(cum_sum_eigenvalues):
    plt.step(
        range(0, len(cum_sum_eigenvalues)),
        cum_sum_eigenvalues,
        where="mid",
        label="Cumulative explained variance",
    )
    plt.ylabel("Explained variance ratio")
    plt.xlabel("Principal component index")
    plt.legend(loc="best")
    #plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) != 2:
        sys.exit("Il faut un csv en argument.")
    loader = FileLoader()
    file = loader.load(sys.argv[1])
    corpus, classes = get_data(file)
    matrix = get_matrix(corpus)
    cum_sum_eigenvalues = calculate_PCA(matrix)
    plot_var_ratio(cum_sum_eigenvalues)


if __name__ == "__main__":
    main()
