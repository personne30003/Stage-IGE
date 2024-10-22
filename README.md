# Fichiers de travail
Ce repertoire regroupe tous les programmes utilisés pendant mon stage.   
Pour les données voir [ce dépôt Google Drive](https://drive.google.com/drive/folders/18Ba65oPThXsAqQ-eXRMZ9NYaX64OMVZB?usp=sharing) 

## Notebooks Python : 
- `Grachev_T2_TO5.ipynb` : Tracé de profils verticaux sur T2, mesure de la hauteur du maximum de vent catabatique $z_{max}$ (inspiré de *[Grachev et al. 2016]*). Visualisation des spectres/cospectres de Fourier en période catabatique, essai de la méthode Bulk
  **A FAIRE : Certaines parties du codes sont redondantes, donc pour raisons de lisibilité ce serait bien d'écrire des fonctions à la place**
- `SEB_net_T2R.ipynb` : Mesure du bilan d'énergie en surface, calcul de la fonte à l'aide du radiomètre et du niveau 1 de T2R.
- `Stabilite_T2_TO5-V2.ipynb` *nom à changer !* : Calcul des spectres MRD, mesure du gap spectral sur T2 (sur données brutes). **Fonctionne bien mais mal écrit. Utiliser de préférence la version suivante (quand elle sera finie)**
- `Stabilite_T2_TO5-V3.ipynb` *nom à changer !* : Idem mais basé intégralement sur `xarray`, donc plus rapide et plus sûr que la version précédente
  **A FAIRE : Finir de l'écrire. C'est pas très compliqué et ne devrait pas prendre trop de temps**
- `Visualisation_snowfox_T2.ipynb` : Visualisation des données Basse-Fréquence (radiomètre, thermomètre CS215, cf tableau 1 de mon rapport)
  **A FAIRE (facultatif) : Remplacer le dictionnaire `series` par un `DataFrame` `pandas`. Parce que là c'est moche et pas robuste**
- `Variabilite_flux.ipynb` : Comparaison des flux turbulents entre les stations T2, T2R et T2L. Mesure de $z_{max}$ sur T2R et T2L (deux niveaux seulement).
- `flux_T2_Left.ipynb` : Distribution des paramètres de stabilité ($R_b$ et $\frac{z}{L}$), comparaison des flux calculés sur 2min et 30min
- `flux_T2_Right.ipynb` : Idem, avec test de la méthode Bulk, dépendance des flux vis à vis de la vitesse du vent et de la température. **Il s'agit du seul notebook où on a le flux de chaleur latente !**.
-  `flux_T2_TO5.ipynb` : Idem que `flux_T2_Left.ipynb`, vérification des lois de similitude (flux gradient et flux-variance), contrôles qualitéq.
-  `rose_des_vents_IGE.ipynb` : Tracé de la rose des vents, à partir de la station météo située sur le toit du bâtiment OSUG-B. Peut toujours servir...
-  `comparaison_flux_brut_vs_EC.ipynb` : Comparaison des flux turbulents/grandeurs moyennes avec/sans prétraitement, sur T2 et sur des segments de 30min.
## Librairies (dont je suis l'auteur)
- `MRD_vX.py` : Implémentation de l'algorithme de MRD, et détection du gap spectral, cf *[Vickers and Mahrt 2003]*.
  Selon les versions, l'algorithme de détection du gap spectral est modifié. Les exemples associés sont dans `MRD_vX_test.py`/`MRD_vX_test.ipynb`. Le jeu de données utilisé est disponible [ici](http://servdap.legi.grenoble-inp.fr/opendap/hyrax/meige/22_TP_TMA/TP09_CLA_MONTAGNE/DATA_2023_3oct/sonicdata_2023_03_10.nc.dmr.html) (il s'agit d'un TP de M2).

  **A FAIRE : écrire une version définitive (et documentée) avec un exemple. De nombreuses fonctions sont inutiles et doivent être supprimées. Rajouter les fonctions écrites dans `Stabilité_T2_TO5-V2.ipynb` (tracé de spectres, utilisation de Xarray). !** 

- `Bulk.py` : Calcul des flux de chaleurs turbulents à l'aide de la méthode du Bulk aérodynamique (cf thèse de Maxime Litt, p55), calcul des rugosités de surface à partir des flux obtenus par EddyCovariance (à l'aide des formules précedentes). Quelques fonctions statistiques aussi ($RMSE$, $R^2$, etc...).
  **A FAIRE : Documenter !!**

## Programmes utilitaires
- `fusion_dat_to_netcdf.py` : Regroupement des fichiers de données brutes en un seul fichier NETCDF4. Utilisé pour les calculs de MRD.
  **Instructions dans les commentaires en début de programme !**
- `read_EddyPro_Output_v2.py` : Lecture des fichiers de sortie d'EddyPro (`full_output`, spectres, ogives), récupération de certaines variables dans d'autres fichiers de sortie ($\langle u'w'\rangle$ dans fichier `fluxnet` par exemple), conversion en fichiers NETCDF.
  **Instructions dans les commentaires en début de programme ! Le code fonctionne bien mais mérite d'être amélioré.**

## Infos en Vrac
Les anémomètres CSAT3B sont (en principe) alignés de façon anti-parallèle vent dominant. Concrètement, sur l'Hintereisferner, le vent souffle vers le Nord-Est donc, vers le bas de la pente. Les anémomètres sont donc (en principe) orientés vers le haut de la pente, c'est à dire vers le Sud-Ouest. En pratique, ils sont décalés de quelques dizaines degrés.

Dans `tower2_log.pdf`, les directions ne se rapportent pas au Nord mais au Sud. Vu que j'ai tout rentré dans EddyPro sans réfléchir, les valeurs de direction du vent étaient décalées de 180 degrés par rapport aux valeurs "vraies". On corrige les valeurs de cette façon : 

```python
#correction manuelle pour la direction du vent
T2_1_2min=T2_1_2min.assign(wind_dir=(T2_1_2min['wind_dir']+180)%360)
```

Il me semble que pour T2R et T2L, j'ai oublié de demander à EddyPro de sortir un fichier de sortie `metadata`. Les hauteurs des instruments sont intégrées de cette façon : 
```python
z_1_30min = np.array([1.2 if date < np.datetime64("2023-09-06 09:00:00") else 1.05
                        for date in T2R_1_30min.coords['temps'].values])
z_T2_1_30min=xr.DataArray(data=z_1_30min,
                         dims=['temps'],
                         coords={'temps':('temps',T2R_1_30min.coords['temps'].values)})
T2R_1_30min=T2R_1_30min.assign({'instrument_height':z_T2_1_30min})
```
C'est pas très beau, mais ça fonctionne.

## Librairies nécessaires : 
Tous les programmes sont écrits en langage **Python 3**. A part la bibliothèque standard, les librairies suivantes sont utilisés : 
- Calcul scientifique de base : [`numpy`](https://numpy.org/install/)/[`scipy`](https://scipy.org/install/)/[`matplotlib`](https://matplotlib.org/stable/).
- Manipulation de données : [`pandas`](https://pandas.pydata.org/docs/getting_started/install.html), [`xarray`](https://docs.xarray.dev/en/stable/getting-started-guide/installing.html). Ici, j'utilise surtout `xarray` ([documentation très bien faite](https://docs.xarray.dev/en/stable/index.html) !)
- Parseur de chaines de caractère : [`scanf`](https://pypi.org/project/scanf/).
- Barres de progression : [`tqdm`](https://tqdm.github.io/)
- Tracé de roses des vents : [`windrose`](https://python-windrose.github.io/windrose/)
- Lecture/écriture de fichiers NETCDF [`netcdf4`](https://unidata.github.io/netcdf4-python/)
Pour les notebooks Python, j'utilise [Jupyter Notebook](https://jupyter.org/install).
Pour les programmes et les librairies, j'utilise soit IDLE (installé de base en même temps que Python), soit [Pycharm](https://www.jetbrains.com/fr-fr/pycharm/).
## Conditions d'utilisation
Faites ce que vous voulez de ces programmes, à deux détails près :
- Pas d'utilisation commerciale (même si je gagne tout/une partie des bénéfices !). 
- Pas d'utilisation à des fins malveillantes.

L'auteur décline toute responsabilité en cas de dommages infligés au matériel ou à des personnes tierces.
