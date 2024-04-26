import sys
import pandas as pd


def get_stats(csv_file):
    df = pd.read_csv(csv_file, sep="|")
    categories = list(set(df["dynamique_type"].to_list()))
    for categorie in categories:
        print(f"Nombre d'elements dans `{categorie}` : {len(df.query('`dynamique_type` == @categorie'))}")
    #print(categories)


def main():
    csv_file = sys.argv[1]
    get_stats(csv_file)


if __name__ == "__main__":
    main()
