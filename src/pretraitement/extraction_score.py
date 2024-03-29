import pandas as pd
import sys
import matplotlib.pyplot as plt


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
    data = df[["Texte", "Pol positive", "Pol negative"]].query(
        "not(`Pol negative` == 0 and `Pol positive` == 0)"
    )
    corpus = data["Texte"]
    score = data["Pol positive"] - data["Pol negative"]
    min_score = score.min()
    max_score = score.max()
    scores_rescaled = pd.DataFrame([((val - min_score) * 5 / (max_score - min_score)) for val in score], columns=["Score"])
    # scores_rescaled.plot(kind="density", title="Densité des scores de polarité")
    # plt.show()
    # print(scores_rescaled)
    final = pd.concat([corpus, scores_rescaled["Score"]], axis=1)
    final.to_csv("corpus_valeurs_continues.csv", index=False)
    return final


def main():
    if len(sys.argv) != 2:
        sys.exit("Il faut un csv en argument.")
    loader = FileLoader()
    file = loader.load(sys.argv[1])
    score = get_data(file)
    loader.display(score, -30)


if __name__ == "__main__":
    main()
