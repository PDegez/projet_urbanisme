#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 13:32:11 2024

@author: pauline
"""
from dataclasses import dataclass
import re
import argparse
from pathlib import Path
import csv
import pickle
import numpy as np
# import spacy


##############################  DATACLASSES  #################################

@dataclass
class Phrase:
    brut : str
    text : str
    ndyn : int 
    npos : int
    nneg : int


@dataclass 
class Fichier:
    nom : str
    phrases : list[Phrase]


@dataclass
class Corpus:
    fichiers : list[Fichier]
    
    
##############################  FONCTIONS  ##################################

def main(dossier=None, output_file=None, output_format=None):
    """Fonction principale : extraire le corpus, analyser le corpus et
    sauvegarder le corpus (optionnel)"""
    
#   nlp = spacy.load("fr_core_news_sm")
    corpus = extraction(dossier)
    corpus_analyse = analyse(corpus)
      
    if output_file and output_format:
        save_format[output_format](corpus_analyse, output_file)
    
    return print("Fin du script")
        

def extraction(dossier: Path): 
    """Créer pour chaque fichier enrichi du corpus un dictionnaire qui au nom 
    du fichier fait correspondre une liste de phrases"""
    fichiers = []
    
    for sub_folder in dossier.iterdir():
        if sub_folder.is_dir() and re.search("Enrichi",sub_folder.name):
            for fichier in sub_folder.iterdir():
                with open(fichier,"r", encoding="utf-8") as file:
                    fichier_txt = file.read()
                    fichier_txt = fichier_txt.replace(" M. ", " M ")
                    fichier_phrases = re.findall("(.{3}.*?[\.\!\?])\s",
                                                 fichier_txt, re.DOTALL)
#                    doc = nlp(fichier_txt)
#                    fichier_phrases = [sent.text for sent in doc.sents]
                    dictionnaire = {fichier.name: fichier_phrases}
                    fichiers.append(dictionnaire)
                     
    return fichiers


def analyse(corpus)-> Corpus:
    """Créer un object Corpus contenant une liste d'objets Fichier.
    Chaque objet Fichier contient un nom et une liste d'objets Phrase.
    Chaque objet Phrase :
        - brut : le texte brut, avec les balises
        - texte : le texte nettoyé, sans les balises
        - ndyn : le nombre de dynamiques présentes dans la phrase
        - npos : le nombre de polarités positives présentes dans la phrase
        - nneg : le nombre de polarités négatives présentes dans la phrase"""
        
    corpus_obj = []
    
    for fichier in corpus:
        for fichier_nom in fichier.keys():
            phrases = fichier[fichier_nom]
            phrases_obj = []
            
            for phrase in phrases :
                brut = phrase.lstrip("1234567890).:;, ")
                text = re.sub("<.*?>", "", brut).lstrip("1234567890).:;, ")
                ndyn = len(re.findall("<dyn", brut))
                npos = len(re.findall("pol\s?=\s?[\"\']pos[\"\']", brut,
                                      re.IGNORECASE))
                nneg = len(re.findall("pol\s?=\s?[\"\']neg[\"\']", brut,
                                      re.IGNORECASE))
                phrases_obj.append(Phrase(brut, text, ndyn, npos, nneg))
            
            fichier_obj = Fichier(fichier_nom, phrases_obj)
            corpus_obj.append(fichier_obj)
    
    corpus_analyse = Corpus(corpus_obj)
    
    nb_phrase = [len(fichier.phrases) for fichier in corpus_analyse.fichiers]
    nb_dyn = [phrase.ndyn for fichier in corpus_analyse.fichiers 
                  for phrase in fichier.phrases]
    nb_pos = [phrase.npos for fichier in corpus_analyse.fichiers 
                  for phrase in fichier.phrases]
    nb_neg = [phrase.nneg for fichier in corpus_analyse.fichiers 
                  for phrase in fichier.phrases]
    
    print("INFORMATIONS CORPUS:\n")
    print(f"Nb total de phrase : {np.sum(nb_phrase)}")
    print(f"Nb de dynamique : {np.sum(nb_dyn)}")
    print(f"Nb de positive : {np.sum(nb_pos)}")
    print(f"Nb de negative : {np.sum(nb_neg)}\n")
    
    
    
#    for phrase in corpus_analyse.fichiers[1].phrases:
#        print(phrase.brut)
#        print(phrase.text, "\n")
    
    return corpus_analyse
                

def save_pickle(corpus: Corpus, output_file: Path) -> None:
    """Enregistrer le corpus au format pickle"""
    
    with open(output_file, "wb") as output_stream:
        pickle.dump(corpus, output_stream)

    return print(f"le fichier pickle a bien été crée au chemin {output_file}")


def save_csv(corpus: Corpus, output_file: Path) -> None:
    """Enregistrer le corpus dans un csv"""
    
    with open(output_file, mode='a', newline='') as fichier_csv:
        writer = csv.writer(fichier_csv)
        
        writer.writerow([
                         "Fichier",
                         "Brut",
                         "Texte",
                         "Dynamique",
                         "Pol positive",
                         "Pol negative"
                         ])
        
        for fichier in corpus.fichiers:
            for phrase in fichier.phrases:
                writer.writerow([
                    fichier.nom,
                    phrase.brut,
                    phrase.text,
                    phrase.ndyn,
                    phrase.npos,
                    phrase.nneg])
                
    return print(f"Le fichier csv a bien été créé au chemin {output_file}") 


save_format = {
    "pickle": save_pickle,
    "csv": save_csv
    }


##############################  SCRIPT  ##################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="extraction et analyse du corpus enrichi")
    parser.add_argument("input_directory", help="Directory 'urbanisme'")
    parser.add_argument("-f", "--output_format", 
        choices=["csv", "pickle"],
        help="Output format : csv or pickle'")    
    parser.add_argument("-o", "--output_file", type=str, help="Output file")
    args = parser.parse_args()
    
    main(Path(args.input_directory), args.output_file, args.output_format)