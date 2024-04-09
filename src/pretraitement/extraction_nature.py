#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 11:23:43 2024

@author: pauline
"""

from dataclasses import dataclass
import re
import argparse
from pathlib import Path
import csv


##############################  DATACLASSES  #################################

@dataclass
class Dynamique:
    brut : str
    text : str
    nature : str


@dataclass 
class Fichier:
    nom : str
    text : str
    dyns : list[Dynamique]


@dataclass
class Corpus:
    fichiers : list[Fichier]
    
    
##############################  FONCTIONS  ##################################

def main(dossier=None, output_file=None, nature_multiple=None):
    """Fonction principale : extraire les dynamiques et leur nature, et les
    sauvegarder (optionnel)"""
    
    corpus = extraction(dossier)
    corpus_analyse = analyse(corpus, nature_multiple)
    save_csv(corpus_analyse, output_file)
    
    return print("Fin du script")
        

def extraction(dossier: Path)-> list[dict]: 
    """Extrait les fichiers sous formes d'une liste de dictionnaire :
            -> clé : nom du fichier
            -> item : contenu txt du fichier en string"""
    fichiers = []
    
    for sub_folder in dossier.iterdir():
        if sub_folder.is_dir() and re.search("Enrichi",sub_folder.name):
            for fichier in sub_folder.iterdir():
                with open(fichier,"r", encoding="utf-8") as file:
                    fichier_txt = file.read()
                    dictionnaire = {fichier.name: fichier_txt}
                    fichiers.append(dictionnaire)
                     
    return fichiers


def analyse(corpus:list[dict], nature_multiple:bool=None)-> Corpus:
    """ Extraction des dynamiques et de leur nature, sortie en un objet corpus"""
    
    fichiers = []
    for fichier in corpus :
        for key, item in fichier.items():
            dyn_brut = re.findall("<dyn.*?/dyn>",item, re.DOTALL)
            dyn_obj = [] 

            for dynamique in dyn_brut:
                texte_dynamique = re.sub("<.*?>", "", dynamique, re.DOTALL)
                if re.search("nature=\"", dynamique):
                    nature_dynamique = re.search("nature=\".*?\"", dynamique).group()
                    nature_dynamique = nature_dynamique.replace("nature=", "")
                    nature_dynamique = nature_dynamique[1:-2]
                    if not args.multiple_nature :
                        if len(nature_dynamique)<=4 :        
                            dyn_obj.append(
                                Dynamique(
                                    dynamique,
                                    texte_dynamique,
                                    nature_dynamique
                                    )
                                )
                    else:
                        dyn_obj.append(
                            Dynamique(
                                dynamique,
                                texte_dynamique,
                                nature_dynamique
                                )
                            )
                        
        fichiers.append(Fichier(key, item, dyn_obj))
        
    corpus_analyse = Corpus(fichiers)
    return corpus_analyse


def save_csv(corpus: Corpus, output_file: Path) -> None:
    """Enregistrer le corpus dans un csv"""
    
    with open(output_file, mode='a', newline='') as fichier_csv:
        writer = csv.writer(fichier_csv, delimiter = "|")
        
        writer.writerow([
                         "fichier",
                         "dynamique_brut",
                         "dynamique_text",
                         "dynamique_type"
                         ])
        
        for fichier in corpus.fichiers:
            for dynamique in fichier.dyns:
                writer.writerow([
                    fichier.nom,
                    dynamique.brut,
                    dynamique.text,
                    dynamique.nature])
                
    return print(f"Le fichier csv a bien été créé au chemin {output_file}") 

##############################  SCRIPT  ##################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="extraction et analyse du corpus enrichi - nature")
    parser.add_argument("input_directory", help="Directory 'urbanisme'")   
    parser.add_argument("output_file", help="Output file")
    parser.add_argument("-m", "--multiple_nature", action="store_true",
                        help="Conserver les natures multiples")
    
    args = parser.parse_args()
    
    main(Path(args.input_directory), args.output_file, args.multiple_nature)