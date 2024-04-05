import pandas as pd
import stanza
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize


class Lemmatizer:
    def __init__(self, stopwords_file):
        self.nlp = stanza.Pipeline(lang="fr", processors="tokenize,lemma")
        self.stopwords = self.get_stopwords(stopwords_file)
    
    def get_stopwords(self,stopwords_file):
        print("opening file")
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


class Stemmer:
    def __init__(self):
        self.stemmer = SnowballStemmer("french")
    
    def stem(self,text):
        tokens = word_tokenize(text)
        roots = [self.stemmer.stem(token)for token in tokens]
        return " ".join(roots)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", help="Fichier contenant le corpus à lemmatiser")
    # parser.add_argument("stopwords_file", help="Fichier contenant les stopwords")
    args = parser.parse_args()
    df = pd.read_csv(args.corpus,sep="|")
    stemmer = Stemmer()
    df["Texte"] = df["Texte"].apply(stemmer.stem)
    df.to_csv("root_corpus.csv", index=False,sep="|")


if __name__ == "__main__":
    main()
