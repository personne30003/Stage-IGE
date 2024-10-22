"""
Module pour le calcul du spectre MRD (Multi-Resolution-Decomposition) et la détermination du gap spectral
Algorithme basé sur la thèse de Sébastien Blein et surtout sur l'article de Vickers et Mahrt, "The co-spectral gap and turbulent flux calculation", 2003,
Auteur : Evan LEMUR 
A FAIRE URGEMMENT : -Mettre à jour les fonctions sur la recherche de gap spectral (cf notebook exemple)
                    -Ecrire d'autres fonctions pour tracer les spectres (cf notebook exemple)
                    -Eventuellement, écrire une fonction de recherche du gap spectral à partir de l'algo présenté dans Voronovich 2007.
"""
import sys
import pandas as pd
import numpy as np
import scipy
import copy
import matplotlib.pyplot as plt
import matplotlib as mpl
try:#ça on peut s'en passer mais dans le cas où quelqu'un d'autre utilise ce code ...
    import tqdm
    import tqdm.notebook
except ModuleNotFoundError:
    print("module tqdm non trouvé")
    print("Ne pas utiliser MRD_segments avec progress_bar=True")

#fonctions pour travailler en base 2
#Bon tutoriel (en C mais c'est pareil) : 
#https://skyduino.wordpress.com/2013/04/05/tuto-le-bitwise-pour-les-nuls/
def MSB(val):#donne le bit de poids fort (c'est à dire le plus à gauche)
    flag=False
    nb_bits=sys.getsizeof(val)#cette fonction renvoie la taille de la variable en bits 
    masque=(1<<nb_bits)
    i=nb_bits
    while flag != True:
        val_masque=(val & masque)>>i
        #print(val_masque)
        if val_masque == True:
            flag=True
        else: 
            i=i-1
            masque=(1<<i)
            #print(bin(masque))
            #print(i)
    return i

def decomposition(n):#fonction pour décomposer un nombre  sous la forme n=2^m+r, où r est le reste, retourne les deux termes
    MSB_n=MSB(n)
    return 2**MSB_n,n-2**MSB_n

def filtre_121(x):
#inspiré de https://dsp.stackexchange.com/questions/48603/what-is-so-special-about-this-filter-coef-1-2-1 (c'est la seule source d'infos que j'ai trouvé)
    res=np.copy(x)
    res[1]=(x[1]+2.0*x[0])/4
    for i in range(2,x.size):
        res[i]=(x[i]+2.0*x[i-1]+x[i-2])/4
    return res

def MRD_serie(serie_1,serie_2,tmin,duree=60,freq_ech=1.0,**kwargs):#duree en minutes
    "A FINIR !!! (PAS URGENT CAR LA V1 MARCHE BIEN ET JE M'EN SERS PAS ALORS BON) "
    duree_h=np.timedelta64(duree,'min')
    tmax=tmin+duree_h
    x=serie_1.truncate(before=tmin,after=tmax).values
    y=serie_2.truncate(before=tmin,after=tmax).values
    res_MRD=MRD(x,y,**kwargs)
    res_MRD['temps']=echelles_temps(res_MRD[0]['co_spectre'].size,freq_ech)
    return res_MRD

def echelles_temps(N,freq_ech=1.0):
    #équivalent de fftfreq dans scipy,N:nombre de points dans le spectre, freq_ech:fréquence d'échantillonnage
    return ((freq_ech)**-1)*np.geomspace(1,2**(N-1),num=N)
def echelles_temps_MSB(N,freq_ech=1.0):
    #idem mais avec N le nombre de points dans un segment. Usage interne seulement.
    MSB_N=MSB(N)
    return ((freq_ech)**-1)*np.geomspace(1,2**(MSB_N-1),num=MSB_N)
def add_segment(dic,dic_segment,ignore='temps'):
    #simple fonction pour concaténer un dictionnaire sur certaines clés
    #utilisée plus bas
    #dic: dictionnaire à concaténer, new_dic: dictionnaire à ajouter (de la même forme que dic), ignore:clé à ignorer 
    new_dic=copy.deepcopy(dic)
    #print(f"dic_segments.keys() {dic_segment.keys()}")
    for key in dic.keys():
        #print(f"key in dic.keys() {key}")
        if key==ignore:
            pass
        else:
            #new_dic[key]=np.stack((new_dic[key],dic_segment[key]))
            new_dic[key].append(dic_segment[key])
    return new_dic

def MRD(x,y,name=('x','y'),get_spectres=True,filtrage=False,get_timescale=True,f_ech=1.0):
    "implémentation simple de l'algo de la publication"
    #x, y : tableaux 1D de même taille
    if (x.size !=y.size):
        raise ValueError(f"x et y n'ont pas la même taille : x.size={x.size}, y.size={y.size}")
    if (len(x.shape) or len(y.shape)) > 1:
        raise ValueError(f"x  et y ne sont pas des tableaux 1D : x.shape ={x.shape}, y.shape={y.shape}")
    #print(f"filtrage {filtrage}")
    #print(f"get_spectres {get_spectres}")
    #print(f"get_timescale {get_timescale}")
    N_points=x.size
    M=MSB(N_points)
    index_mult,reste=decomposition(N_points)
    x_i=np.copy(x)[:index_mult]
    y_i=np.copy(y)[:index_mult]
    x_rm=x_i-np.nanmean(x_i)
    y_rm=y_i-np.nanmean(y_i)
    co_spectre=np.zeros(M)
    spectre_x=np.zeros(M)
    spectre_y=np.zeros(M)
    for m in range(M-1,-1,-1):
        n=2**(M-m)
        x_split_m=np.array_split(x_rm,n)
        y_split_m=np.array_split(y_rm,n)
        x_rm_inter=np.copy(x_split_m)
        y_rm_inter=np.copy(y_split_m)
        x_split_moy=np.nanmean(x_split_m,axis=1)
        y_split_moy=np.nanmean(y_split_m,axis=1)
        for i in range(0,n):
            x_rm_inter[i]=x_rm_inter[i]-x_split_moy[i]
            y_rm_inter[i]=y_rm_inter[i]-y_split_moy[i]
        spectre_x[m]=np.nanvar(x_split_moy)
        spectre_y[m]=np.nanvar(y_split_moy)
        co_spectre[m]=np.nanmean((x_split_moy-np.nanmean(x_split_moy))*(y_split_moy-np.nanmean(y_split_moy)))
        x_rm=np.ravel(x_rm_inter)
        y_rm=np.ravel(y_rm_inter)
    #filtrage
    if filtrage==True:
        co_spectre=filtre_121(co_spectre)
        spectre_x=filtre_121(spectre_x)
        spectre_y=filtre_121(spectre_y)
    #calcul du flux
    flux=np.cumsum(co_spectre)
    spectre1_cumul=np.cumsum(spectre_x)
    spectre2_cumul=np.cumsum(spectre_y)
    res={}
    res[name[0]+name[1]]=co_spectre
    res[name[0]+name[1]+'_cum']=flux
    if get_timescale==True:
        res['temps']=echelles_temps(M,f_ech)
    if get_spectres==True:
        spectres={}
        spectres[name[0]]=spectre_x
        spectres[name[0]+'_cum']=spectre1_cumul
        spectres[name[1]]=spectre_y
        spectres[name[1]+'_cum']=spectre2_cumul
        if get_timescale==True:
            spectres['temps']=echelles_temps(M,f_ech)
        return res,spectres
    else : return res


def MRD_segments(x, y,name=('x','y'),filtrage=False,f_ech=1.0,get_spectres=True,bar_progress=False,notebook=True):
    #x et y sont cette fois des tableaux 2D (matrices rectangulaires)
    # retourne un dictionnaire de la même forme que celui retourné par MRD, mais avec, pour chaque clé  (sauf échelles de temps) un tableau 2D
    #avec les spectres et les flux pour chaque segments
    #bar_progress=False -> pas de barre de progression (par defaut)
    #notebook=True -> on est sous jupyter notebook,
    # A FAIRE : si le dernier segment comporte trop de NANs, lancer un Warning
    if x.shape !=y.shape:raise ValueError(f"x and y have not the same shape {x.shape} and {y.shape}")
    shape=x.shape
    if len(shape) < 2 :raise ValueError("1D array. please use MRD(x,y,**kwargs) function ...")
    N_segments=shape[0]
    taille_segments=shape[1]
    temps=echelles_temps_MSB(taille_segments,f_ech)
    #print(f"N_segments {N_segments}")
    #print(f"taille_segments {taille_segments}")
    #print(f"temps {temps}")
    res={'temps':temps,(name[0]+name[1]):[],(name[0]+name[1]+'_cum'):[]}
    spectres={}
    if get_spectres==True:
        spectres={'temps':temps,name[0]:[],name[0]+'_cum':[],name[1]:[],name[1]+'_cum':[]}
    else:
        del spectres
    iterateur_segments=range(0,N_segments)
    if bar_progress==True:
        iterateur_segments=tqdm.notebook.tqdm(iterateur_segments) if notebook==True else tqdm.tqdm(iterateur_segments)
    for i in iterateur_segments:
        MRD_i=MRD(x[i],y[i],name,get_spectres,filtrage,get_timescale=False)
        #print(f"MRD_i {MRD_i}")
        #attention, ce qui suit est horrible (bouts de codes trop redondants)
        #a l'avenir, faudra faire une fonction (FAIT)
        if get_spectres==True:
            res=add_segment(res,MRD_i[0])
            spectres=add_segment(spectres,MRD_i[1])
        else:
            res=add_segment(res,MRD_i)
    if get_spectres==True:
        return res, spectres
    else:
        return res

def spectre_moy(res_MRD):
    #calcule la moyenne et l'écart type d'un spectre MRD
    #remplace les clé principales (ex:"flux" , "co_spectre") par la moyenne, ajoute une clé supplémentaire
    #de la forme cle_principale+'_err'
    new_res=copy.deepcopy(res_MRD)
    for key in res_MRD.keys():
        if key=='temps':
            pass
        else:
            new_res[key]=np.mean(res_MRD[key],axis=0)
            new_res[key+'_err']=np.std(res_MRD[key],axis=0)
    return new_res
def spectre_median(res_MRD):
    #idem que ci-dessus mais avec la médiane et l'écart inter-quartile
    new_res=copy.deepcopy(res_MRD)
    for key in res_MRD.keys():
        if key=='temps':
            pass
        else:
            new_res[key]=np.median(res_MRD[key],axis=0)
            new_res[key+'_err']=scipy.stats.iqr(res_MRD[key],axis=0)
    return new_res
def plot_spectre(res_MRD,var,ax=None,mode='moy',**kwargs):
    #fonction qui trace le spectre moyenné avec barres d'erreur
    #res_MRD: dictionnaire avec spectre+barres d'erreur, var:variable à tracer,*args et**kwargs: paramètres à passer dans plt.bar(genre color,label etc..)
    #c'est juste pour les faignants, aucune obligation de l'utiliser
    #A renommer en plot_spectre_moy
    if isinstance(ax,mpl.axes.Axes):
        ax.errorbar(res_MRD['temps'],res_MRD[var],yerr=res_MRD[var+'_err'],**kwargs)
    else:
        plt.errorbar(res_MRD['temps'],res_MRD[var],yerr=res_MRD[var+'_err'],**kwargs)

def plot_spectre_segments(res_MRD,var,mode='moy',ax=None,xlabel=None,ylabel=None):
    #trace le spectre moyen/median avec les barres d'erreur, en noir  et les autres spectres en fond
    #paramètres :-pour les deux premiers : idem que plot_spectre
    #            -mode : le spectre en noir est soit le spectre moyen (mode='moy'), soit le spectre mode='med'
    #            -ax : dans le cas où la figure comporte de multiples tracés, il s'agit simplement du subplot dans lequel on se place
    #            -xlabel,ylabel : labels sur axes x et y
    func_plot=plt.plot#A FAIRE : remplacer ces fonctions par un dictionnaire
    func_xlabel=plt.xlabel
    func_ylabel=plt.ylabel
    func_legend=plt.legend
    func_xscale=plt.xscale
    func_error_bar=plt.errorbar
    if isinstance(ax,mpl.axes.Axes):
        func_plot=ax.plot
        func_xlabel = ax.set_xlabel
        func_ylabel = ax.set_ylabel
        func_xscale = ax.set_xscale
        func_error_bar=ax.errorbar
    fonc_spectre_principal={'moy':spectre_moy,'med':spectre_median}
    for segments in res_MRD[var]:
        func_plot(res_MRD['temps'],segments,color='tab:gray',linestyle='dotted',alpha=0.5)
    res_MRD_err=fonc_spectre_principal[mode](res_MRD)
    func_error_bar(res_MRD_err['temps'],res_MRD_err[var],yerr=res_MRD_err[var+'_err'],ecolor='k',color='k',capthick=10)
    func_xscale('log')
    func_ylabel(ylabel)
    func_xlabel(xlabel)
    return res_MRD_err

def gap_spectral_moy(res_MRD,var,**kwargs):
    #algo qui consiste à calculer le gap spectral sur tous les segments puis à moyenner
    #comme dans la publication. res_MRD : cette fois, c'est un dictionnaire avec les segments
    #donc les valeurs sont des tableaux 2D
    liste_gap_spectral=[]
    liste_pic_spectral=[]
    N_segments=len(res_MRD[var])
    for i in range(0,N_segments):
        pic,gap=gap_spectral_algo(res_MRD[var][i],res_MRD[var+'_cum'][i],res_MRD['temps'],**kwargs)
        liste_gap_spectral.append(gap)
        liste_pic_spectral.append(pic)
    return np.nanmean(liste_pic_spectral),np.nanmean(liste_gap_spectral)
def gap_spectral(res_MRD,var,**kwargs):
    "idem que gap_spectral_moy mais pour un seul spectre"
    return gap_spectral_algo(res_MRD[var],res_MRD[var+'_cum'],res_MRD['temps'],**kwargs)
def gap_spectral_algo(spectre,spectre_sum,index,seuil=0.01,amplitude_pic=0.2,t_min=400):
    "retourne le gap spectral et le premier pic (associé à la turbulence)"
    #algorithme basé sur celui présenté dans Vickers and Mahrt 2003.
    #le pic associé à la turbulence est supposé être le plus fort en amplitude
    #le gap spectral est obtenu lorsqu'après avoir atteint le pic associé à la turbulence, le spectre cumulé varie
    # de moins de 1%
    #res_MRD : dictionnaire contenant le spectre et le spectre cumulé,var:variable à traiter, index : échelles de temps
    #Je garantis pas que ça donnera des résultats systématiquement cohérents
    gap=0.0
    spectre_copy=np.copy(spectre)
    #print(f"spectre_copy {spectre_copy}")
    spectre_sum_copy=np.copy(spectre_sum)
    arg_pics,caracteristiques=scipy.signal.find_peaks(np.abs(spectre),prominence=amplitude_pic*np.abs(spectre).max())
    if arg_pics.size==0: return np.NaN,np.NaN
    arg_pic_spectre=arg_pics[0]
    #print(index[arg_pic_spectre])
    if index[arg_pic_spectre] > t_min:return np.NaN,np.NaN
    pic=index[arg_pic_spectre]
    pic=np.around(index[arg_pic_spectre],3)
    gap=0.0
    #iterateur=range(arg_pic,len(spectre_copy))
    for i in range(arg_pic_spectre,len(spectre_copy)):
        #delta_spectre_sum=(spectre_sum[i+1]-spectre_sum[i-1])/2.0#shéma centré
        #delta_spectre_sum=(spectre_sum[i+1]-spectre_sum[i])#shéma vers la droite
        delta_spectre_sum=(spectre_sum[i]-spectre_sum[i-1])#shéma vers la gauche
        if (np.abs(delta_spectre_sum/spectre_sum[i]) <= seuil):
            gap=index[i]
            return pic,np.around(gap,3)
    return pic,np.NaN

def signe_pic(array):
    "détermine le signe du plus grand pic d'un spectre"
    min_array = np.min(array)
    max_array = np.max(array)
    return -1 if (np.abs(min_array) > np.abs(max_array)) else 1
