import pandas as pd
import sys

class FileLoader:
    def load(self, path):
        df = pd.read_csv(path)
        rows, columns = df.shape
        print(f"Loading dataset of dimensions {rows} x {columns}")
        return df

    def display(self, df, n=None):
        if n is not None:
            if n >= 0:
                print(df.head(n))
            else:
                print(df.tail(-n))
        else:
            print(df)


def get_data(df):
    data = df[["Texte", "Pol positive", "Pol negative"]]

    pol_positive = data["Pol positive"] > data["Pol negative"]
    pol_negative = data["Pol negative"] > data["Pol positive"]
    phrases_pos = data[pol_positive]
    phrases_neg = data[pol_negative]
    corpus_pos = phrases_pos["Texte"].to_list()
    corpus_neg = phrases_neg["Texte"].to_list()
    corpus = corpus_pos + corpus_neg
    classes = ["Positive"] * len(corpus_pos) + ["Negative"] * len(corpus_neg)
    print(len(corpus), len(classes))


def main():
    if len(sys.argv) != 2:
        sys.exit("Il faut un csv en argument.")
    loader = FileLoader()
    file = loader.load(sys.argv[1])
    get_data(file)


if __name__ == "__main__":
    main()
