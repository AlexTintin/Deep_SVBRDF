from __future__ import print_function
import time
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
#matplotlib inline
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import os
from PIL import Image
import numpy as np
import random

random.seed(0)
np.random.seed(0)


def get_statistics_all(path_folder_images):
    """
    path_folder_images : chemin vers les images directement

    output : dictionnaire avec en clef1 le matériel principale, en clef 2 le matériel secondaire
    (séparé par 'X' dans le nom) et en valeur une liste de type 0X3.png
    """

    all_files = os.listdir(path_folder_images)
    dico = {}
    r=0
    for i in range(len(all_files)):
        file1 = all_files[i]
        file1_list = file1.split(";")
        new_key = file1_list[1].split("X")[0]
        key2 = file1_list[1].split("X")[1]
        if new_key not in dico.keys():
            r+=1
            dico[new_key] = {}
            dico[new_key][key2] = [file1_list[-1],file1_list[0]]
        else:
            if key2 not in dico[new_key].keys():
                dico[new_key][key2] = [file1_list[-1],file1_list[0]]
            else:
                dico[new_key][key2].append(file1_list[-1])
                dico[new_key][key2].append(file1_list[0])
    return (dico,r)

path_folder_images = "../DeepMaterialsData/train"
dico_all,rty = get_statistics_all(path_folder_images)
#print(dico_all)

r = np.zeros([1,rty],int)
compteur=0
liste = []
liste1=[]
for index_key, key in enumerate(dico_all.keys()):
    if index_key%7==0:
        liste1.append(key)
for index_key, key in enumerate(dico_all.keys()):
    for ale in range(len(liste1)):
        if liste1[ale] == key :
            for index_key2, key2 in enumerate(dico_all[key].keys()):

                for al in range(len(liste1)):
                    if liste1[al] == key2:
                        a = []

                        a.append(dico_all[key][key2][0])
                        a.append(dico_all[key][key2][1])
                        for k in range(int(len(dico_all[key][key2])/2)):
                            b = dico_all[key][key2][2*k]
                            l=0
                            for i in range(int(len(a)/2)):
                                if a[2*i]==b:
                                    l+=1
                            if l==0:
                                a.append(b)
                                a.append(dico_all[key][key2][2*k+1])
                        for i in range(int(len(a)/2)):
                            liste.append(a[2*i+1]+';'+key+'X'+key2+';'+a[2*i])


print(len(liste))


os.system('mkdir ../theones_20/')
for i in range(len(liste)):
    im = Image.open(path_folder_images + '/' + liste[i])
    im.save('../theones_20/' + liste[i])
    print(i)

"""

namesfilestokeeptrain = []

for index_key, key in enumerate(dico_all.keys()):
    compteurb = 0
    randompick = np.sort(np.random.randint(0,int(r[0,index_key]/2),1,int))
    i=0
    for index_key2, key2 in enumerate(dico_all[key].keys()):
        for i in range(1):
            if(compteurb<=randompick[i]):
                cbn = compteurb + int(len(dico_all[key][key2])/2)
                if (cbn > randompick[i]):
                    namesfilestokeeptrain.append(dico_all[key][key2][2*(randompick[i]-compteurb)+1]+';'+key+'X'+key2+';'+dico_all[key][key2][2*(randompick[i]-compteurb)])
        compteurb += int(len(dico_all[key][key2])/2)

print(len(namesfilestokeeptrain))

#os.system('mkdir ../new_dataset')
os.system('mkdir ../new_dataset/val')
#for i in range(len(namesfilestokeeptrain)):
#   os.system('cp ../DeepMaterialsData/train/'+namesfilestokeeptrain[i]+' ../new_dataset/train/' )
#print(path_folder_images+'/'+namesfilestokeeptrain[i])
for i in range(len(namesfilestokeeptrain)):
    im = Image.open(path_folder_images+'/'+namesfilestokeeptrain[i])
    im.save('../new_dataset/val/'+namesfilestokeeptrain[i])
    print(i)
"""