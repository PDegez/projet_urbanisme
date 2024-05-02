import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np


PATH_SA = "Data/Results_polarity_models_sentiment_analysis.csv"
PATH_CAMEMBERT = "Data/Results_polarity_models_camembert.csv"


def get_labels():
    df = pd.read_csv(PATH_CAMEMBERT, sep="|").dropna()
    labels = df["class"]
    return list(labels)


def get_camembert_preds():
    preds = pd.read_csv(PATH_CAMEMBERT, sep="|")["distilcamembert"]
    preds = [int(pred[0]) for pred in preds if not isinstance(pred, float)]
    preds = ["Positive" if pred > 3 else "Negative" if pred < 3 else "Neutral" for pred in preds]
    return preds

def get_sa_preds():
    preds = pd.read_csv(PATH_SA, sep="|")["distilcamembert"]
    preds = ["Positive" if pred == "positive" else "Negative" if pred == "negative" else "Neutral" if pred == "neutral" else None for pred in preds]
    preds = [pred for pred in preds if pred is not None]
    print(len(preds))
    return preds


def handle_neutral(classes, preds):
    new_preds = []
    for i, pred in enumerate(preds):
        if pred == "Neutral":
            if classes[i] == "Positive":
                new_preds.append("Negative")
            else:
                new_preds.append("Positive")
        else:
            new_preds.append(pred)
    return new_preds


def generate_confusion_matrix(prediction, classes):
    conf_matrix = confusion_matrix(classes, prediction)
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=np.unique(classes), )
    disp.plot(cmap="OrRd")
    plt.show()


def print_prediction(prediction, classes):
    report = str(classification_report(classes, prediction))
    print(report)


def main():
    classes = get_labels()
    sa_preds = get_sa_preds()
    #camembert_preds = get_camembert_preds()
    #camembert_preds = handle_neutral(classes, camembert_preds)
    sa_preds = handle_neutral(classes, sa_preds)
    #print(classes)
    #print(len(camembert_preds))
    generate_confusion_matrix(sa_preds, classes)
    #generate_confusion_matrix(camembert_preds, classes)
    #print_prediction(camembert_preds, classes)
    print_prediction(sa_preds, classes)



if __name__ == "__main__":
    main()
