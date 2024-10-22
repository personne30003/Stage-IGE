"""
fonctions pour la méthode Bulk. On utilisera z/L calculé à partir d'EddyPro
dans un premier temps
A faire : La fonction wq semble donner des résultats incohérents. Régler ça.
MAJ : Aucun problème en fait.
"""
import numpy as np
import xarray as xr
import scipy
import matplotlib as mpl
import matplotlib.pyplot as plt


kappa=0.4
#beta_m=4.7
#beta_h=4.7
#beta_q=4.7
#z_0=0.001#m
#z_t=0.01*z_0
#z_q=0.01*z_qP

def P_sat(T):
    "retourne la pression de vapeur saturante en Pa (cf Charrondiere), pour une température T en K"
    return 611.0*np.exp(((2.5e6)/462.0)*(1.0/273.15-1.0/T))

def q_sat(T,P):
    "retourne l'humidité spécifique à saturation (RH=100%)."
    #T : K, P: Pa
    return 0.622*P_sat(T)/P

#chaleur latente de vaporisation (en J/kg
#Tair en K
L_v=lambda Tair:1.0e3*(3147.5-2.37*Tair)

def u_s(u_moy,
        z,
        zL,
        beta_m=4.7,
        z0=0.001):
    print("appel u_s()")
    print(f"z0 {z0}")
    return (kappa*u_moy)/(np.log(z/z0)+beta_m*zL)

def wT(T_moy,
       u_moy,
       z,
       zL,
       T_s=273.15,
       zt=0.00001,
       beta_t=4.7,
       **kwargs):
    print("appel wT")
    print(f"zt {zt}")
    return -u_s(u_moy,z,zL,**kwargs)*(kappa*(T_moy-T_s))/(np.log(z/zt)+beta_t*zL)

def wq(q_moy,
       u_moy,
       z,
       zL,
       Press,
       T_s=273.15,
       zq=0.00001,
       beta_q=4.7,
       **kwargs):
    print("appel wq")
    print(f"zq {zq}")
    return -u_s(u_moy,z,zL,**kwargs)*(kappa*(q_moy-q_sat(T_s, Press)))/(np.log(z/zq)+beta_q*zL)

#fonctions plus haut niveau
#pour utiliser les Datasets
def us_bulk(Ds,**params):
    return u_s(Ds['u_rot'],
               Ds['instrument_height'],
               Ds['zL'],
               **params)

def wT_bulk(Ds,**params):
    return wT(Ds['air_temperature'],
              Ds['u_rot'],
              Ds['instrument_height'],
              Ds['zL'],
              **params)
def wq_bulk(Ds,**params):
    return wq(Ds['specific_humidity'],
              Ds['u_rot'],
              Ds['instrument_height'],
              Ds['zL'],
              Ds['air_pressure'],
             **params)

def H_bulk(Ds,**params):
    return Ds['air_heat_capacity']*wT_bulk(Ds,**params)

def LE_bulk(Ds,**params):
    return (L_v(Ds['air_temperature'])*Ds['air_density'])*wq_bulk(Ds,**params)

#fonctions pour calculer la rugosité de surface à partir des flux turbulents, sur un seul niveau
#D'après Fitzpatrick 2019
def get_z0(u_moy,u_star,z,zL,beta_m=4.7):
    return np.exp(-beta_m*zL-kappa*(u_moy/u_star))*z

def get_zt(T_moy,T_star,z,zL,T_s=273.15,beta_h=4.7):
    return np.exp(-beta_h*zL-kappa*(T_moy-T_s)/T_star)*z

def get_zq(q_moy,wq,u_star,Press,z,zL,T_s=273.15,beta_q=4.7):
    qs=q_sat(T_s,Press)
    q_star=-wq/u_star
    return np.exp(-beta_q*zL-kappa*(q_moy-qs)/q_star)*z

#idem : fonctions de plus haut niveau
def z0_EC(Ds,**params):
    return get_z0(Ds['u_rot'],
                  Ds['u*'],
                  Ds['instrument_height'],
                  Ds['zL'],
                  **params)
def zt_EC(Ds,**params):
    return get_zt(Ds['air_temperature'],
                  Ds['T*'],
                  Ds['instrument_height'],
                  Ds['zL'],
                  **params)
def zq_EC(Ds,**params):
    return get_zq(Ds['specific_humidity'],
                  Ds['wh2o_cov'],
                  Ds['u*'],
                  Ds['air_pressure'],
                  Ds['instrument_height'],
                  Ds['zL'],
                  **params)

#Quelques fonctions utiles
#nombre de Richardson Bulk
def Ri_B(U,T_z,z,T_s=273.15):
    "retourne le nombre de Richardson Bulk, avec T_z et T_s les températures à la hauteur z et en surface (en K)"
    return (9.81*(T_z-T_s)*z)/(T_s*(U**2))

def get_Ri_B(Ds,**params):
    return get_Ri_B(Ds['u_rot'],
                    Ds['air_temperature'],
                    Ds['instrument_height'],
                    **params)

def slope_1(x,**kwargs):
    x_range=np.linspace(np.nanmin(x),np.nanmax(x),x.size)
    plt.plot(x_range,x_range,**kwargs)

def R_2(model,data):
    #retourne le coef. de correlation R^2 entre deux tableaux
    masque= np.array(np.isnan(model) | np.isnan(data))#on enlève les NaN
    new_model=np.ma.array(model,mask=masque).compressed()
    new_data=np.ma.array(data,mask=masque).compressed()
    return scipy.stats.linregress(new_model,new_data).rvalue**2

#fonctions pour évaluer le modèle par rapport aux données.

def MBE(model,data):
    return np.nanmean(model-data)
def abs_MBE(model,data):
    return np.nanmean(np.abs(model-data))

def MSE(model,data):
    return np.nanmean((data-model)**2)

def MBE_norm(model,data):
    return MBE(model,data)/np.mean(data)


def RMSE(model,data):
    return np.sqrt(MSE(data,model))

def RMSE_norm(model,data):
    return np.sqrt(MSE(data,model))/np.nanstd(data)

def MRBE(model, data):
    return np.mean(np.abs((data-model)/data))

def texte(model,data,units="W/m^2"):
    R2_texte="$R^2$ = {:.3f}".format(R_2(model, data))+"\n"
    MBE_texte="MBE = {:.3f}".format(MBE(model, data))+" "+units+"\n"
    MRBE_texte="MRBE = {:.3f} %\n".format(100.0*MRBE(model, data))
    RMSE_texte="RMSE = {:.3f}".format(RMSE(model, data))+" "+units
    return R2_texte+MBE_texte+MRBE_texte+RMSE_texte