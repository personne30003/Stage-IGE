"""
Script permettant de récupérer les données de sorties d'EddyPro : spectres, ogives, fichiers full_output
et certaines méta-données (hauteur de l'instrument). Les Datasets sont convertis en fichiers NETCDF4 (oui, je suis amoureux de ce format).

Entrées:
         -dossier_output_NC => Dossier où les fichiers NETCDF4 de de sortie seront créés
         -dic_projet => dictionnaire de la forme {repertoire_projet_EddyPro:nom}. "repertoire_projet_EddyPro" est le répertoire où se trouvent les fichiers de sortie d'EddyPro.
         "nom" est simplement le nom du fichier qui sera créé par ce programme (voir plus bas).

Sorties : 
         -fichier [nom]_full_output.nc => fichier full_output d'EddyPro, avec en plus quelques variables supplémentaires comme 'instrument_height (m)' 
         (si fichier metadata détecté), 'wu_cov' (<u'w'>, pris à partir du fichier fluxnet). Les variables avec un nom comprenant un '/' ou un '%' sont
         renommées (les noms de variables ne peuvent pas contenir ces caractères). Ainsi, '(z-d)/L' devient 'zL', 'w/ts_cov' devient 'wts_cov', 'x_90%' devient 'x_90pc'.
         -fichier [nom]_spectres.nc => tous les spectres sont rassemblés en un seul Dataset 2D (une dimension de temps, une dimension de fréquence). Les noms de variables contenant un
         '/' sont renommées. Ainsi 'f_nat*spec(u)/var(u)' devient 'f_Su_norm', 'f_nat*cospec(w_u)/cov(w_u)' devient 'f_C_wu_norm'. En complément, les spectres de la vitesse et 
         de la température sont 'dé-normalisés'. Cela peut servir par exemple pour calculer le taux de dissipation d'énergie. 
         -fichiers [nom]_ogives.nc => idem que les spectres mais avec les ogives.

Note : Vu qu'il y a une barre de progression pour les spectres, il est fortement conseillé de NE PAS faire tourner ce code sous IDLE. Le résultat sera moche !

Dépendances (modules hors bibliothèque standard): numpy, pandas, xarray, scanf, tqdm.
Pour installer ces modules, ouvrir un shell, et taper la commande pip install [nom du module]. Si pip n'est pas installé, eh bien ... il faut l'installer !

Ce code ne marche que pour Python 3. 

A FAIRE : -récupérer le contenu des fichiers "qc_detail" (pour avoir les détails des contrôles qualité), et le mettre dans le fichier full_output. C'est facile à faire, donc
          pourquoi s'en priver ?
          -Afficher le temps d'éxecution restant (faire des moyennes glissantes).
          -Accélérer le temps d'éxecution, même si je vois pas trop comment faire, vu que je supprime toutes mes variables après qu'elles aient servi (et de toute
          façon, le "garbage collector" s'en charge tout seul). A mon avis, on peut pas faire grand chose, ça a probablement à voir avec l'OS.
          -IMPORTANT : Fusionner fichiers spectres/ogives et full_output ensemble ! Demande un peu de travail mais ça devrait pas être trop compliqué
MAJ le 11/07/2024 : ajout de la fonction main (dans laquelle se déroule le programme)
Auteur: Evan LEMUR
"""

import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
import scanf
import copy
import time as clk
import warnings
# module pour avoir barre de progression (c'est stylé !)
# détecte automatiquement le type de shell utilisé (ipython, standard ...)
from tqdm.auto import tqdm
# ci-dessous les seuls paramètres à modifier...
dossier_output_NC = r"C:\Users\evanl\Documents\Stage IGE\Outputs_EddyPro"

dic_projets = {
    #T2_TO5 2min
    #r"C:\Users\evanl\Documents\Stage IGE\T2_TO5_EddyPro\output-1-2min":"T2_TO5_1_2min",
    #r"C:\Users\evanl\Documents\Stage IGE\T2_TO5_EddyPro\output-1-30min-new":"T2_TO5_1_30min-new",
    #r"C:\Users\evanl\Documents\Stage IGE\T2_TO5_EddyPro\output-2-30min-new":"T2_TO5_2_30min-new",
    #r"C:\Users\evanl\Documents\Stage IGE\T2_TO5_EddyPro\output-3-30min-new":"T2_TO5_3_30min-new"
    #T2_TO5_30min
    #r"C:\Users\evanl\Documents\Stage IGE\T2_TO5_EddyPro\output-1-30min":"T2_TO5_1_30min",
    #r"C:\Users\evanl\Documents\Stage IGE\T2_TO5_EddyPro\output-2-30min":"T2_TO5_2_30min",
    #r"C:\Users\evanl\Documents\Stage IGE\T2_TO5_EddyPro\output-3-30min":"T2_TO5_3_30min",
    #T2 Right 30min
    #r"C:\Users\evanl\Documents\Stage IGE\stations\tower2-right-east\output-haut-30min":"T2_RE_haut_30min",
    r"C:\Users\evanl\Documents\Stage IGE\stations\tower2-right-east\output-bas-30min_all_final":"T2_RE_bas_30min_all_final"
    #r"C:\Users\evanl\Documents\Stage IGE\stations\tower2-right-east\output-bas-30min":"T2_RE_bas_30min",
    #T2 Right 2min
    #r"C:\Users\evanl\Documents\Stage IGE\stations\tower2-right-east\output-haut-2min":"T2_RE_haut_2min",
    #r"C:\Users\evanl\Documents\Stage IGE\stations\tower2-right-east\output-bas-2min":"T2_RE_bas_2min",
    #T2 Left 30min
    #r"C:\Users\evanl\Documents\Stage IGE\stations\tower2-left-west\output-bas-30min":"T2_LW_bas_30min",
    #r"C:\Users\evanl\Documents\Stage IGE\stations\tower2-left-west\output-haut-30min":"T2_LW_haut_30min",
    #T2 Left 2min
    #r"C:\Users\evanl\Documents\Stage IGE\stations\tower2-left-west\output-bas-2min":"T2_LW_bas_2min",
    #r"C:\Users\evanl\Documents\Stage IGE\stations\tower2-left-west\output-haut-2min":"T2_LW_haut_2min"
}


def parse_date(nom_fichier,format_nom,ordre={'y':0,'M':1,'d':2,'h':3,'m':4}):
    "fonction qui extrait la date d'un nom de fichier. A améliorer"
    #print(f"ordre {ordre}")
    scan_date = scanf.scanf(format_nom,nom_fichier)
    #print(f"scan_date {scan_date}")
    new_ordre = {}
    if scan_date == None:
        raise IndexError(f"mauvais format de nom : {nom_fichier}, {format_nom}")
    if len(scan_date) <len(ordre.values()):raise IndexError
    for key in ordre.keys():#permet de mettre un nombre sous la forme 0u où u est l'unité
        #print(ordre[key])
        digit_date=scan_date[int(ordre[key])]
        new_ordre[key]='0'+str(digit_date) if digit_date<10 else str(digit_date)
    date_str="{}-{}-{} {}:{}:00".format(*tuple(new_ordre.values()))
    return pd.to_datetime(date_str)

def open_Eddypro_full_output(nom_fichier):
    "renvoie un dataset xarray à partir d'un fichier de sortie full_output"
    Df_flux=pd.read_csv(nom_fichier,header=1,
                        usecols=lambda x: x not in ['filename','DOY','file_records','used_records'],
                        skiprows=lambda x: x in [2])
    datetime=pd.to_datetime(Df_flux['date'].values+' '+Df_flux['time'].values)
    Df_flux=Df_flux.assign(temps=datetime).drop(columns=["date","time"])
    Ds_flux=Df_flux.to_xarray().assign_coords({'index':Df_flux['temps'].values}).drop_vars('temps')
    dic_rename={'index':'temps',
                'w/ts_cov':'wts_cov',
                'x_10%':'x_10pc',
                'x_30%':'x_30pc',
                'x_50%':'x_50pc',
                'x_90%':'x_90pc',
                'x_70%':'x_70pc',
                '(z-d)/L':'zL'}
    if 'w/h2o_cov' in Ds_flux.data_vars:
        dic_rename['w/h2o_cov']='wh2o_cov'
    if 'w/ch4_cov' in Ds_flux.data_vars:
        dic_rename['w/ch4_cov']='wch4_cov'
    if 'w/co2_cov' in Ds_flux.data_vars:
        dic_rename['w/co2_cov']='wco2_cov'
    if 'w/none_cov' in Ds_flux.data_vars:
        dic_rename['w/none_cov']='wnone_cov'
    Ds_flux=Ds_flux.rename(dic_rename)
    return Ds_flux.drop_vars(var_full_nan(Ds_flux))

def parse_date_fluxnet(date_columns):
    format_date="%4d%2d%2d%2d%2d"
    return np.array([parse_date(str(date_columns[i]),format_date) for i in range(0,len(date_columns))])

def open_EddyPro_fluxnet(nom_fichier):
    "renvoie un dataset xarray à partir d'un fichier de sortie fluxnet"
    Df_flux=pd.read_csv(nom_fichier)
    datetime=parse_date_fluxnet(Df_flux['TIMESTAMP_END'].values)
    #print(f"datetime : {datetime}")
    Df_flux=Df_flux.drop(columns=["TIMESTAMP_START","FILENAME_HF"])
    Ds_flux=Df_flux.to_xarray().assign_coords({'index':datetime}).rename({'index':'temps'})
    return Ds_flux


def var_full_nan(Ds):
    "renvoie une liste comprenant toutes les variables pleines de NaN"
    var_nan = []
    full_nan = lambda x: np.isnan(x).all()
    for variable in Ds.data_vars:
        if full_nan(Ds[variable].values) == True:
            var_nan.append(variable)
    return var_nan


def create_spectra_dataset(fichier, date):
    """renvoie un dataset, à une date donnée,
    dont toutes les variables vides (remplies avec des NaN sont retirées"""
    Df_spectre = pd.read_csv(fichier, header=11, usecols=lambda x: x != ' #_freq')
    Df_spectre = Df_spectre.apply(lambda x: x.replace(-9999.0, np.NaN))
    Ds_spectre = Df_spectre.to_xarray()
    Ds_spectre = Ds_spectre.assign_coords(
        {"index": Ds_spectre["natural_frequency"].values}).rename({'index': 'freq'})
    # Dans la ligne suivante,je renomme les variables (les caractères '/' sont pas tolérés lors
    #                                                 de la conversion au fichier NETCDF).
    Ds_spectre = Ds_spectre.rename({'f_nat*spec(u)/var(u)': 'f_Su_norm',
                                    'f_nat*spec(v)/var(v)': 'f_Sv_norm',
                                    'f_nat*spec(w)/var(w)': 'f_Sw_norm',
                                    'f_nat*spec(ts)/var(ts)': 'f_Sts_norm',
                                    'f_nat*spec(h2o)/var(h2o)': 'f_Sh2o_norm',
                                    'f_nat*spec(co2)/var(co2)': 'f_Sco2_norm',
                                    'f_nat*spec(ch4)/var(ch4)': 'f_Sch4_norm',
                                    'f_nat*spec(none)/var(none)': 'f_Snone_norm',
                                    'f_nat*cospec(w_u)/cov(w_u)': 'f_C_wu_norm',
                                    'f_nat*cospec(w_v)/cov(w_v)': 'f_C_wv_norm',
                                    'f_nat*cospec(w_ts)/cov(w_ts)': 'f_C_wts_norm',
                                    'f_nat*cospec(w_co2)/cov(w_co2)': 'f_C_wco2_norm',
                                    'f_nat*cospec(w_h2o)/cov(w_h2o)': 'f_C_wh2o_norm',
                                    'f_nat*cospec(w_ch4)/cov(w_ch4)': 'f_C_w_ch4_norm',
                                    'f_nat*cospec(w_none)/cov(w_none)': 'f_C_w_none_norm'})

    # Ds_spectre=Ds_spectre.drop_vars(['index','normalized_frequency','natural_frequency'])
    Ds_spectre = Ds_spectre.drop_vars(['natural_frequency'])
    #Ds_spectre = Ds_spectre.drop_vars(var_full_nan(Ds_spectre))
    Ds_spectre = Ds_spectre.expand_dims(dim={"temps": [date]}, axis=1)
    return Ds_spectre

def open_EddyPro_metadata_file(fichier):
    Df_flux = pd.read_csv(fichier)
    datetime = pd.to_datetime(Df_flux['date'].values+' '+Df_flux['time'].values)
    Df_flux = Df_flux.assign(temps=datetime).drop(columns=["date","time"])
    Ds_flux = Df_flux.to_xarray().assign_coords({'index':Df_flux['temps'].values}).drop_vars('temps')
    return Ds_flux.rename({'index':'temps'})


def create_ogive_dataset(fichier, date):
    "idem que create_spectra_dataset mais avec les ogives"
    Df_spectre = pd.read_csv(fichier, header=11, usecols=lambda x: x != ' #_freq')
    Df_spectre = Df_spectre.apply(lambda x: x.replace(-9999.0, np.NaN))
    Ds_spectre = Df_spectre.to_xarray()
    Ds_spectre = Ds_spectre.assign_coords(
        {"index": Ds_spectre["natural_frequency"].values}).rename({'index': 'freq'})
    # Dans la ligne suivante,je renomme les variables (les caractères '/' sont pas tolérés lors
    #                                                 de la conversion au fichier NETCDF).
    Ds_spectre = Ds_spectre.rename({'og(u)': 'og_u',
                                    'og(v)': 'og_v',
                                    'og(w)': 'og_w',
                                    'og(ts)': 'og_ts',
                                    'og(co2)': 'og_co2',
                                    'og(h2o)': 'og_h2o',
                                    'og(ch4)': 'og_none',
                                    'og(w_u)': 'og_wu',
                                    'og(w_v)': 'og_wv',
                                    'og(w_ts)': 'og_wts',
                                    'og(w_co2)': 'og_wco2',
                                    'og(w_h2o)': 'og_wh2o',
                                    'og(w_ch4)': 'og_wch4',
                                    'og(w_none)': 'og_wnone', })

    # Ds_spectre=Ds_spectre.drop_vars(['index','normalized_frequency','natural_frequency'])
    Ds_spectre = Ds_spectre.drop_vars(['natural_frequency'])
    #Ds_spectre = Ds_spectre.drop_vars(var_full_nan(Ds_spectre))
    Ds_spectre = Ds_spectre.expand_dims(dim={"temps": [date]}, axis=1)
    return Ds_spectre

def read_files_spectra(dossier,func_create_dataset):
    """fonction utilisée pour lire les fichiers de spectres et d'ogives
       renvoie un dataset xarray"""
    liste_Ds = []
    forme_fichiers = "%4d%2d%2d-%2d%2d%*s"
    for fichiers in tqdm(os.listdir(dossier)):
        date_fichier = parse_date(fichiers,forme_fichiers)
        Ds_fichier = func_create_dataset(dossier+fichiers,date_fichier)
        liste_Ds.append(Ds_fichier)
    Ds_tot = xr.concat(liste_Ds,dim='temps',join='override')
    return Ds_tot.sortby('temps').drop_vars(var_full_nan(Ds_tot))

def xarray_to_netcdf(Ds,nom_fichier):
    try:
        Ds.to_netcdf(nom_fichier+'.nc')
    except FileExistsError:
        warnings.warn("erreur : le fichier {} existe déjà".format(nom_fichier),RuntimeWarning)

def read_EddyPro_folder(dossier_EddyPro_output,dossier_outputs,nom_projet):
    """convertit les spectres/ogives et les fichiers full_output EddyPro en fichiers NC
    A FAIRE : - Traiter exceptions PandasMultiIndex (ne la traite pas en détail, affiche juste un Warning pour dire qu'elle
                est présente, et met le flag à False)
              - Voir si tu peux pas rassembler les spectres et les ogives dans un seul fichier
              - Rendre le code plus léger en le découpant en fonctions, là ça devient une HORREUR à lire !
    """
    try :
        liste_fichiers = os.listdir(dossier_EddyPro_output)
    except FileNotFoundError:
        warnings.warn("erreur : le chemin d'accès spécifié est introuvable : {}".format(dossier_EddyPro_output),
                     RuntimeWarning)
        print("terminé")
        return
    flag_ogive = False
    flag_cospectre = False
    flag_full_output = False
    flag_metadata = False
    flag_fluxnet = False

    Ds_ogive =None
    Ds_cospectre = None
    Ds_metadata = None
    Ds_full_output = None
    Ds_fluxnet = None
    t1=clk.time()
    for fichier in liste_fichiers:
        if 'binned_ogives' in fichier:
            try:
                flag_ogive = True
                print("données ogives détectées")
                print("extraction")
                dossier_ogive = dossier_EddyPro_output+r"\eddypro_binned_ogives"+r"\\"
                Ds_ogive = read_files_spectra(dossier_ogive,create_ogive_dataset)
                print("fait")
            except Exception as err:
                warnings.warn("erreur {}".format(err.args), RuntimeWarning)
                flag_ogive=False
        if 'binned_cospectra' in fichier:
            try:
                flag_cospectre = True
                print("données spectre détectées")
                print("extraction")
                dossier_cospectre = dossier_EddyPro_output+r"\eddypro_binned_cospectra"+r"\\"
                Ds_cospectre = read_files_spectra(dossier_cospectre,create_spectra_dataset)
                print("fait")
            except Exception as err:
                warnings.warn("erreur {}".format(err.args), RuntimeWarning)
                flag_cospectre=False
        if 'full_output' in fichier:
            flag_full_output = True
            print("fichier full output détecté")
            print("extraction")
            fichier_full_output = dossier_EddyPro_output+r'\\'+fichier
            Ds_full_output = open_Eddypro_full_output(fichier_full_output)
            print("fait")
        if '_metadata_' in fichier:
            flag_metadata = True
            print("fichier metadata détecté")
            print("extraction")
            fichier_metadata = dossier_EddyPro_output+r'\\'+fichier
            Ds_metadata = open_EddyPro_metadata_file(fichier_metadata)
            print("fait")
            #extraction
        if 'fluxnet' in fichier:
            flag_fluxnet = True
            print("fichier fluxnet détecté")
            print("extraction")
            fichier_fluxnet = dossier_EddyPro_output+r'\\'+fichier
            Ds_fluxnet = open_EddyPro_fluxnet(fichier_fluxnet)
            print("fait")
        #extraction
    if not ((flag_full_output and flag_metadata and flag_fluxnet) or flag_ogive or flag_cospectre):
        print("aucun fichier de sortie d'EddyPro détecté")
        print("au revoir comme disait l'autre")
        return
    if flag_cospectre:
        if flag_full_output:
            Ds_cospectre = Ds_cospectre.assign({'f_Su': Ds_cospectre["f_Su_norm"] * Ds_full_output['u_var'],
                                                'f_Sv': Ds_cospectre["f_Sv_norm"] * Ds_full_output['v_var'],
                                                'f_Sw': Ds_cospectre["f_Sw_norm"] * Ds_full_output['w_var'],
                                                'f_Sts': Ds_cospectre["f_Sts_norm"] * Ds_full_output['ts_var'],
                                                'f_C_wts': Ds_cospectre["f_C_wts_norm"] * Ds_full_output['wts_cov']})
        print("conversion des spectres en un fichier NETCDF")
        t2 = clk.time()
        #Ds_cospectre.to_netcdf(dossier_output+r"\\"+id_output+"_spectres.nc")
        xarray_to_netcdf(Ds_cospectre,dossier_outputs + r"\\" + nom_projet + "_spectres")
        print(f"fait en {clk.time()-t2} s")
    if flag_ogive:
        print("conversion des ogives en un fichier NETCDF")
        t2 = clk.time()
        #Ds_ogive.to_netcdf(dossier_output + r"\\" + id_output + "_ogives.nc")
        xarray_to_netcdf(Ds_ogive, dossier_outputs + r"\\" + nom_projet + "_ogives")
        print(f"fait en {clk.time() - t2} s")
    del Ds_ogive
    del Ds_cospectre
    if flag_full_output:
        print("conversion du fichier full_output au format NETCDF4")
        t2 = clk.time()
        if flag_metadata:
            Ds_instrument_height=xr.Dataset(data_vars={'instrument_height' :(Ds_full_output.dims,
                                                                             Ds_metadata['master_sonic_height'].values)},
                                            coords=Ds_full_output.coords)
            Ds_full_output = xr.merge([Ds_full_output,Ds_instrument_height])
            del Ds_instrument_height
            #print(Ds_instrument_height)
        if flag_fluxnet:
            Ds_wu_cov=xr.Dataset(data_vars={'wu_cov' : (Ds_full_output.dims,
                                                        Ds_fluxnet['W_U_COV'].values)},
                                 coords=Ds_fluxnet.coords)
            Ds_full_output = xr.merge([Ds_full_output,Ds_wu_cov])
            Ds_full_output = Ds_full_output.assign(vw_cov=np.sqrt(Ds_full_output['u*']**4-Ds_full_output['wu_cov']**2))
            del Ds_wu_cov
        #print(Ds_wu_cov)
        #problèmes de caractères illégaux dans les noms de variables. On met ful.l_output au format csv pour le moment
        #bon, en fait on s'y fera
        #Ds_full_output.to_netcdf(dossier_output+r"\\"+id_output+"_full_output.nc")
        xarray_to_netcdf(Ds_full_output, dossier_outputs + r"\\" + nom_projet + "_full_output")
        print(f"fait en {clk.time() - t2} s")
    del Ds_full_output
    del Ds_fluxnet
    del Ds_metadata
    print(f"processus terminé en {clk.time()-t1} s")


def main():
    "fonction principale ..."
    print("bonjour " + os.environ.get("USERNAME"))
    liste_projets = list(dic_projets.keys())
    N_projets = len(liste_projets)
    print("dossiers à lire : ")
    [print(f"-{i}") for i in liste_projets]
    print(f"nombre de dossiers: {N_projets}")
    print(f"repertoire de sortie {dossier_output_NC}")
    print("lecture : ")
    cpt = 1
    t1 = clk.time()
    for dossier_EddyPro in liste_projets:
        print(f"{cpt} de {N_projets}")
        print(f"dossier  : {dossier_EddyPro}")
        print(f"nom du projet  : {dic_projets[dossier_EddyPro]}")
        tps_exec_1 = clk.time()
        read_EddyPro_folder(dossier_EddyPro, dossier_output_NC, dic_projets[dossier_EddyPro])
        tps_exec_2 = clk.time()
        tps_exec_estime = int((N_projets - cpt) * (tps_exec_2 - tps_exec_1))
        print(f"temps restant estimé : {tps_exec_estime // 60}:{tps_exec_estime % 60} minutes")
        cpt += 1
    tps_tot = clk.time() - t1
    print(f"processus terminé en {tps_tot // 60}:{tps_tot % 60} minutes")
    print("au revoir !")

if __name__=='__main__':
    main()