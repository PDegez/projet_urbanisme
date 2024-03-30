import pandas as pd
import sys
import matplotlib.pyplot as plt
import stanza


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


class Lemmatizer:
    def __init__(self, stopwords):
        self.nlp = stanza.Pipeline(lang="fr", processors="tokenize,lemma")
        self.stopwords = stopwords

    def lemmatize(self, text):
        doc = self.nlp(text)
        lemmas = [
            word.lemma
            for sentence in doc.sentences
            for word in sentence.words
            if word.lemma not in self.stopwords
        ]
        return " ".join(lemmas)


def parse_stopwords(stopwords_file):
    with open(stopwords_file, "r") as file:
        return file.read().splitlines()

def get_data(df, lemmatizer):
    data = df[["Texte", "Pol positive", "Pol negative"]].query(
        "not(`Pol negative` == 0 and `Pol positive` == 0)"
    )
    corpus = data["Texte"].apply(lemmatizer.lemmatize)
    score = data["Pol positive"] - data["Pol negative"]
    min_score = score.min()
    max_score = score.max()
    scores_rescaled = pd.DataFrame([((val - min_score) * 10 / (max_score - min_score)) for val in score], columns=["Score"])
    corpus = corpus.reset_index(drop=True)
    # scores_rescaled.plot(kind="density", title="Densité des scores de polarité")
    # plt.show()
    final = pd.concat([corpus, scores_rescaled["Score"]], axis=1)
    return final


def main():
    if len(sys.argv) != 3:
        sys.exit("Il faut un csv en argument et des stopwords en argument.")
    loader = FileLoader()
    file = loader.load(sys.argv[1])
    stopwords = parse_stopwords(sys.argv[2])
    lemmatizer = Lemmatizer(stopwords)
    score = get_data(file, lemmatizer)
    score.to_csv("corpus_valeurs_continues.csv", index=False, sep="|")


if __name__ == "__main__":
    main()
