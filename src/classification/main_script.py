#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 11:27:53 2024

@author: pauline
"""

import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from naive_bayes import naive_bayes_classification


class FileLoader:
    
    def load(self, path):
        """transforme le csv en dataframe"""
        
        df = pd.read_csv(path)
        rows, columns = df.shape
        print(f"Loading dataset of dimensions {rows} x {columns}")
        
        return df

    
def get_data(df):
    """Extrait les phrases et leurs classes depuis un dataframe"""
    
    data = df[["Texte", "Classe"]]    
    phrases = data["Texte"]
    classes = data["Classe"]
    
    return phrases, classes
        
  
def get_matrix(phrases):
    """vectorise les phrases"""
    
    vectorizer = TfidfVectorizer(input="content")
    matrix = vectorizer.fit_transform(phrases).toarray()
    
    return matrix
        
################################ MAIN ####################################

def main ():
    parser = argparse.ArgumentParser(description="lanceur de classifieurs")
    parser.add_argument("input_file", type=str, help="Chemin vers le fichier contenant les data csv")
    parser.add_argument("-c", "--classifieur", choices = ["NB", "SVM", "DT", "RF"], help = "Choix du modèle de classifieur")
    args = parser.parse_args()
    
    if args.classifieur:
        
        # Load le corpus et extrait les vecteurs (matrices) et les classes
        loader = FileLoader()
        file = loader.load(args.input_file)
        phrases, classes = get_data(file)  
        matrix = get_matrix(phrases)
        
        # Dictionnaire pour lancer le classifieur en fonction de l'option choisie
        modele = {
            "NB": naive_bayes_classification,
            "SVM": "plop",
            "DT": "plop",
            "RF" : "plop"
            }
        
        # Lancement du classifieur
        modele[args.classifieur](matrix, classes)
        
    else :
        print("Choississez un classifieur : option -c suivi du modele choisi :")
        print("-c NB \tpour lancer Naive Bayes")
        print("-c SVM \tpour lancer SVM")
        print("-c DT \tpour lancer Decision Tree")
        print("-c RF \tpour lancer Random Forest")
     
if __name__ == "__main__":
    main()