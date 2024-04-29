#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 10:20:35 2024

@author: pauline
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv


def parser():
    parser = argparse.ArgumentParser(
        description="Statistiques descriptives du corpus")
    parser.add_argument("-p", "--boxplots",
                        action="store_true", help="Affichage des boxplots")
    parser.add_argument("-c", "--csv", type=str, help="Chemin de l'output csv")
    parser.add_argument("-n", "--nature", action="store_true",
                        help="travail sur le csv Nature")
    parser.add_argument("-b", "--barre", action="store_true",
                        help="diagramme en barre")
    parser.add_argument("input_file",
                        help="Chemin vers le fichier csv contenant le corpus") 
    args = parser.parse_args()
    
    return args


class FileLoader:
    def load(self, path):
        """transforme le csv en dataframe"""

        df = pd.read_csv(path, sep="|")
        rows, columns = df.shape
        print(f"Loading dataset of dimensions {rows} x {columns}")

        return df


def get_data(df, option_n):
    """Extrait les longueurs des phrases par classe depuis un dataframe"""
    
    result = {}  
    if not option_n :
        data = df[["Texte", "Classe"]]
        pos = data.query('`Classe` == "Positive"')["Texte"]
        neg = data.query('`Classe` == "Negative"')["Texte"]
        result["positive"] = [ len(x.split()) for x in pos ]
        result["negative"] = [ len(x.split()) for x in neg ]        
    else:
        data = df[["dynamique_text", "dynamique_type"]]
        result["ab"] = [ len(x.split()) for x in data.query(
            '`dynamique_type` == "ab"')["dynamique_text"] ]
        result["cre"] = [ len(x.split()) for x in data.query(
            '`dynamique_type` == "cre"')["dynamique_text"] ]
        result["dest"] = [ len(x.split()) for x in data.query(
            '`dynamique_type` == "dest"')["dynamique_text"] ]
        result["main"] = [ len(x.split()) for x in data.query(
            '`dynamique_type` == "main"')["dynamique_text"] ]
        result["modi"] = [ len(x.split()) for x in data.query(
            '`dynamique_type` == "modi"')["dynamique_text"] ]
        result["rep"] = [ len(x.split()) for x in data.query(
            '`dynamique_type` == "rep"')["dynamique_text"] ]
    
    return result


def plot_descriptifs(data: dict):
    """affichage des statistiques descriptives des datas d'un dictionnaire dans
    des boxplots"""
    
    # Création d'une liste géante pour pouvoir plus tard exclure les données
    # abérantes
    all_data = np.concatenate(list(data.values()))
    fig, ax = plt.subplots()
    
    # Création des plots et coloration des boites
    boxplot = ax.boxplot(data.values(), patch_artist=True, showmeans=True, widths=0.5)
    
    plt.title('Boxplots')
    plt.xlabel('categories')
    plt.ylabel('Nb de mots')
    
    # Ajustement de la fenêtre pour exclure du visuel les 5% de données supérieures
    # (qui sont abérantes)
    limite_superieure = np.percentile(all_data, 95)
    ax.set_ylim(0, limite_superieure)
    
    # Changement des couleurs
    for box in boxplot['boxes']:
        box.set(facecolor='cyan')
    for median in boxplot['medians']:
        median.set(color='black')
    for mean in boxplot['means']:
        mean.set(color='white') 
    
    # Assignation des clés du dictionnaire aux labels des plots
    ax.set_xticklabels(data.keys())

    plt.show()

def plot_barres(data:dict):

    valeurs = [len(x) for x in data.values()]
    categories = [x for x in data.keys()]
    
    # Creation diagramme en barres
    plt.bar(categories, valeurs, width=0.5)
    
    # Titres + étiquettes
    plt.title('Diagramme en Barres')
    plt.xlabel('categories')
    plt.ylabel('valeurs')

    # Affichage du diagramme
    plt.show()

    
def write_csv(data:dict, chemin:str):
    with open(chemin, mode='a', newline='') as fichier_csv:
        writer = csv.writer(fichier_csv)
    
        writer.writerow(["",
                         "LONGUEUR MOYENNE",
                         "LONGUEUR MAX",
                         "LONGUEUR MIN",
                         "ECART TYPE",
                         "LONGUEUR MEDIANE",
                         "PREMIER QUARTILE",
                         "TROISIEME QUARTILE"])
    
        for categorie in data.keys():
            writer.writerow([
                categorie.upper(),
                np.mean(data[categorie]),
                max(data[categorie]),
                min(data[categorie]),
                np.std(data[categorie]),
                np.mean(data[categorie]),
                np.percentile((data[categorie]), 25),
                np.percentile((data[categorie]), 75),
                ])
            
    return print(f"Le fichier csv a bien été créé au chemin {chemin}") 


def main(fichier=None, plot=None, csv=None, nature=None, barre=None):

    loader = FileLoader()
    file = loader.load(args.input_file)
    result = get_data(file, nature)
    
    #Statistiques vers sortie standard
    print("\nSTATISTIQUES DESCRIPTIVES POUR NOS CATEGORIES : \n")

    for cle in result.keys():
        print(f"CATEGORIE : {cle}")
        print(f"longueur moyenne : {np.mean(result[cle])}")
        print(f"longueur max : {max(result[cle])}")
        print(f"longueur min : {min(result[cle])}")
        print(f"écart type : {np.std(result[cle])}")
        print(f"longueur médiane : {np.percentile(result[cle], 50)}")
        print(f"premier quartile : {np.percentile(result[cle], 25)}")
        print(f"troisième quartile : {np.percentile(result[cle], 75)}\n")
    
    total = np.concatenate(list(result.values()))
    print("\tTOTAL")
    print(f"longueur moyenne : {np.mean(total)}")
    print(f"longueur max : {max(total)}")
    print(f"longueur min : {min(total)}")
    print(f"écart type : {np.std(total)}")
    print(f"longueur médiane : {np.percentile(total, 50)}")
    print(f"premier quartile : {np.percentile(total, 25)}")
    print(f"troisième quartile : {np.percentile(total, 75)}\n")
    
    #Statistiques vers sortie csv
    if csv:
        write_csv(result, csv)
    
    #Diagramme en barre du nombre de dynamiques par categories
    if barre:
        plot_barres(result)
        
    #Statistiques : affichage des plots
    if plot:
        plot_descriptifs(result)


if __name__ == "__main__":
    args=parser()
    print(args)
    main(
        fichier=args.input_file,
        plot=args.boxplots,
        csv=args.csv,
        nature=args.nature,
        barre = args.barre)