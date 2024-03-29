import pandas as pd
import stanza


class Lemmatizer:
    def __init__(self, stopwords_file):
        self.nlp = stanza.Pipeline(lang="fr", processors="tokenize,lemma")
        self.stopwords = self.get_stopwords(stopwords_file)
    
    def get_stopwords(self,stopwords_file):
        with open(stopwords_file, "r") as file:
            return file.read().splitlines()

    def lemmatize(self, text):
        doc = self.nlp(text)
        lemmas = [
            word.lemma
            for sentence in doc.sentences
            for word in sentence.words
            if word.lemma not in self.stopwords
        ]
        return " ".join(lemmas)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", help="Fichier contenant le corpus à lemmatiser")
    parser.add_argument("stopwords_file", help="Fichier contenant les stopwords")
    args = parser.parse_args()
    df = pd.read_csv(args.corpus,sep="|")
    lemmatizer = Lemmatizer(args.stopwords_file)
    df["Texte"] = df["Texte"].apply(lemmatizer.lemmatize)
    df.to_csv("corpus_lemmatized.csv", index=False,sep="|")


if __name__ == "__main__":
    main()
