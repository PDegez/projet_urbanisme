import sys
from pretaitement import get_data, FileLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd


def get_matrix(corpus):
    vectorizer = TfidfVectorizer(input="content", stop_words="english")
    matrix = vectorizer.fit_transform(corpus).toarray()
    return matrix


def get_pca(matrix, classes):
    pca = PCA(n_components=2)
    pca_matrix = pca.fit_transform(matrix)
    output = pd.DataFrame(pca_matrix, columns=["1", "2"])
    classes_df = pd.DataFrame(classes, columns=["Class"])
    output = pd.concat([output, classes_df], axis=1)
    return output


def plot_pca(output):
    plt.figure(figsize=(10, 6))
    classes = output["Class"].unique()
    for class_name in classes:
        color = "black"
        if class_name == "Positive":
            color = "maroon"
        elif class_name == "Negative":
            color = "lightcoral"
        class_data = output[output["Class"] == class_name]
        plt.scatter(
            class_data["1"], class_data["2"], label=class_name, color=color, alpha=0.75
        )
    plt.legend()
    plt.grid()
    plt.title("PCA reduction 2 components")
    plt.show()


def main():
    if len(sys.argv) != 2:
        sys.exit("Il faut un csv en argument.")
    loader = FileLoader()
    file = loader.load(sys.argv[1])
    corpus, classes = get_data(file)
    matrix = get_matrix(corpus)
    pca_vec = get_pca(matrix, classes)
    plot_pca(pca_vec)


if __name__ == "__main__":
    main()
