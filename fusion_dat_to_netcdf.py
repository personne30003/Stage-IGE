"""
!!!!! A LIRE !!!!! :
Fusion des fichiers météo au format DAT en un seul fichier NETCDF4.

PARAMETRES : 

-La variable repertoire_sortie est le répertoire où vous voulez que le fichier final soit créé.
-La variable repertoire_fichiers est celui où se trouvent les données météos (par exemple celles de "tower-left-west").
-La variable nom_fichier_sortie est le nom du fichier créé.
-La variable format_nom est le format du nom des fichiers météo (qui contient la date). Décommenter la ligne associée.
 
DEPENDANCES : 

Ce script utilise la librairie scanf (qui permet d'utiliser la fonction du même nom, en langage C).
Pour l'installer : ouvrir un terminal et taper pip install scanf
Pour installer xarray  et netCDF4: taper pip install xarray et pip install netCDF4 ...

Ce code mérite franchement d'être amélioré MAIS IL MARCHE ...

MAJ: -le dictionnaire "variables" n'est plus nécessaire
     -codage des flottants sur 32 bits au lieu de 64. On diminue ainsi la taille du fichier obtenu
A FAIRE : mettre une liste de variables à ignorer (genre csat_diag), ça fera toujours ça de moins en mémoire.
"""
import os
import numpy as np
import pandas as pd
import netCDF4 as nc
import xarray as xr
import time as clk
import scanf

repertoire_courant=os.getcwd()

#nom_fichier_sortie="tower2-left-west"
nom_fichier_sortie="temp-T2L"
repertoire_sortie=repertoire_courant+'\\'

#repertoire_fichiers=repertoire_courant+r"\stations\tower2-left-west\data"+'\\'#pour "tower2-left-west'
#repertoire_fichiers=repertoire_courant+r"\stations\tower2-right-east\data"+'\\'#pour "tower2-right-east'
#repertoire_fichiers=repertoire_courant+r"\T2_TO5"+'\\'#pour Irgason
repertoire_fichiers=repertoire_courant+r"\stations\temp-tower2-left\jour"+'\\'#pour thermomètres
print(repertoire_courant)
#dictionnaire spécifiant les types des variables lues dans le fichier
"""
variables={'TIMESTAMP':'str','RECORD':np.int64,'csatB_u':np.float64,'csatB_u':np.float64,'csatB_v':np.float64,
           'csatB_w':np.float64,'csatB_t':np.float64,'csatB_diag':np.float64,'csatH_u':np.float64,'csatH_u':np.float64,'csatH_v':np.float64,
           'csatH_w':np.float64,'csatH_t':np.float64,'csatH_diag':np.float64}
"""

#forme d'un nom de fichier. %d désigne un entier et %2d un entier à deux chiffres
#format_nom=r"TOA5_12590.TeamX_csat_%d_%d_%d_%2d%2d.dat"#Pour tower2-left-west
#format_nom=r"TOA5_6079.TeamX_licor_%d_%d_%d_%2d%2d.dat"#pour tower-2-right_east
#format_nom=r"TOA5_TOB1_T2_Wind_50ms_%d_%d_%d_%2d%d_%d_%d_%d_%2d%2d.dat"#Pour Irgason
format_nom=r"TOA5_TOB1_TOA5_8057.fw_pt100_2034_11_13_0106_%d_%d_%d_%2d%2d.dat"#pour thermomètres

var_date='TIMESTAMP'#variable désignant la date ...
#On chargera un dataset avec la commande suivante:
#Df=pd.read_csv(path_fichier1,header=1,skiprows=lambda x: x in [2, 3],low_memory=False)
#Dans la ligne ci-dessus, la fonction lambda indique les lignes du fichier à NE PAS lire 

def create_XrDataset(Df_pd,var_date='TIMESTAMP',var_rec='RECORD'):
    "fonction servant à creer un Dataset Xarray avec une seule dimension temporelle"
    for var in Df_pd.columns:
        if var == var_date:
            Df_pd[var_date]=pd.to_datetime(Df_pd[var_date].apply(lambda x:x.strip("\"")),format='mixed')
        elif var==var_rec:
            Df_pd[var]=Df_pd[var].astype('int64')
        else:
            #Df_pd[var]=Df_pd[var].astype('float64')
            Df_pd[var]=Df_pd[var].astype('float32')#pour réduire la place
    Df_xr=Df_pd.to_xarray()
    #suppression de la variable 'index' comme coordonnée.Néanmoins, elle reste considérée comme une dimension.
    Df_xr=Df_xr.drop_vars('index')
    return Df_xr

def parse_date(nom_fichier,format_nom):
    "fonction qui extrait la date d'un nom de fichier. A améliorer"
    scan_date=scanf.scanf(format_nom,nom_fichier)
    liste_scan=[(lambda x : '0'+str(x) if x <10 else str(x))(scan_date[i]) for i in range(0,5)]#permet de mettre un nombre sous la forme 0u où u est l'unité
    date_str="{}-{}-{} {}:{}:00".format(liste_scan[0],liste_scan[1],liste_scan[2],liste_scan[3],liste_scan[4])
    return np.datetime64(date_str)

def sort_dic_key(dic_date,var_date):
    "fonction permettant de trier un dictionnaire dic_date selon une clé (var_dat)"
    dic_date_sort={}
    arg_sort=np.argsort(dic_date[var_date])
    for var in dic_date:
        dic_date_sort[var]=[dic_date[var][args] for args in arg_sort]
    return dic_date_sort

def get_list_file_date(path,format_nom):
    "lit les noms des fichiers dans le répertoire path et retourne un dictionnaire avec le nom et la date"
    liste_fichiers=os.listdir(path)
    dic_fichier={"name":[],"date":[]}
    for fichiers in liste_fichiers:
        dic_fichier["name"].append(fichiers)
        dic_fichier["date"].append(parse_date(fichiers,format_nom))
    return dic_fichier

print(f"bonjour {os.environ.get("USERNAME")}")
dic_fichiers_repertoire=get_list_file_date(repertoire_fichiers,format_nom)
dic_fichiers_tri=sort_dic_key(dic_fichiers_repertoire,"date")
Ds_liste=[]
t1=clk.time()
for fichiers in dic_fichiers_tri["name"]:
    print(f"lecture de {fichiers}")
    t2=clk.time()
    Df_fichier=pd.read_csv(repertoire_fichiers+fichiers,header=1,skiprows=lambda x: x in [2, 3],low_memory=False)
    Ds_liste.append(create_XrDataset(Df_fichier))
    print(f"fait en {clk.time()-t2} s")
print(f"fichiers chargés en mémoire\n fait en  {clk.time()-t1} s \ncréation d'un Dataset : ")
t1=clk.time()
Ds_xr=xr.concat(Ds_liste,dim='index')
print(f"fait en {clk.time()-t1} s, voici le Dataset obtenu")
print(Ds_xr)
print(f"création d'un fichier au format NETCDF4 dans le répertoire {repertoire_sortie}")
t1=clk.time()
Ds_xr.to_netcdf(repertoire_sortie+nom_fichier_sortie+'.nc')
print(f"fait en {clk.time()-t1} s")
print("voici le fichier obtenu : ")
r=nc.Dataset(repertoire_sortie+nom_fichier_sortie+'.nc','r')
print(r)
print(f"Au revoir {os.environ.get("USERNAME")} !!!")
