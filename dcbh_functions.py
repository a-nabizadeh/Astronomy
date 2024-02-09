import numpy as np
from scipy.stats import chi2



def choose_filters_model(model, obs_num):
    if obs_num == 1:
        filters = model[['F090W', 'F115W','F150W', 'F200W', 'F277W', 'F356W', 'F410M', 'F444W' ]]
        params = model[['Redshift','Mstars']]
        
    return filters,params


def choose_filters_obs(data):
    
    filters_ir = data[['F090W','F115W','F150W','F200W','F277W','F356W','F410M','F444W']]
    filters_err_ir = data[['eF090W','eF115W','eF150W','eF200W','eF277W','eF356W','eF410M','eF444W']]
    
    return filters_ir, filters_err_ir


def convert_to_flux(data):
    flux = data.apply(lambda x: 1e6*3631*10**(x/(-2.5)))
    return flux


def convert_to_ab(data):    #--in muJy
    ab_mag = data.apply(lambda x: -2.5*np.log10(x*1e-6/3631))
    return ab_mag

def offset_fact(observation, observation_error, model):
    offset = np.sum((observation/model)*(observation/observation_error))/np.sum(observation/observation_error)
    return offset

def red_chi2(observation, observation_error, model, offset_factor, model_params):
    Chi2 = np.sum(((observation - offset_factor * model) / (observation_error))**2)
    dof = np.size(observation)-model_params
    red_chi2 = Chi2/dof
    p_value = 1 - chi2.cdf(Chi2, dof)
    return red_chi2, p_value

def survey_limit_JWST(ABMag):
    F_nu = 1e6 * 3631 * 10**(np.array(ABMag) / (-2.5))         #--convert to flux 
    N = F_nu/5                                                 #--1-sigma error on F_nu 
    two_sigma_detection_limit = (2 * N) * 1e3 #nJy             #--The 2-sigma error is then at the 2*N level (S/N=2) upper limit in nJy

    survey_limits_JWST_ABmag = ABMag - 2.5 * np.log10(0.2)     #--JWST: AB mag (5 sigma) limits recomputed to 1 sigma
    survey_limits_JWST_flux = 1e6 * 3631 * 10 ** (survey_limits_JWST_ABmag / (-2.5))  #muJy
    return two_sigma_detection_limit,survey_limits_JWST_flux

def error_calculations(dataframes, flux = True):
    df1_err_cal = dataframes.copy()

    L1 = ['F090W',
    'F115W',
    'F150W',
    'F200W',
    'F277W',
    'F356W',
    'F410M',
    'F444W']

    L2 = ['F090W_up',
    'F115W_up',
    'F150W_up',
    'F200W_up',
    'F277W_up',
    'F356W_up',
    'F410M_up',
    'F444W_up']

    L3 = ['eF090W',
    'eF115W',
    'eF150W',
    'eF200W',
    'eF277W',
    'eF356W',
    'eF410M',
    'eF444W']

    def subtraction(c1,c2):
        return c1 - c2

    def addition(c1,c2):
        return c1 + c2
        
    for i in L3:
        df1_err_cal.loc[df1_err_cal[i] < 0.1, i] = 0.1 

    for i,j,k in zip(L1,L2,L3):
        df1_err_cal[j] = list(map(addition, df1_err_cal[i], df1_err_cal[k]))

        
    df1_err_cal.drop(['ID','RA(deg)','Dec(deg)'], axis=1, inplace=True)

    if flux == False:
        df1_err_cal = convert_to_flux(df1_err_cal)

    for i,j,k in zip(L1,L2,L3):
        df1_err_cal[k] = list(map(subtraction, df1_err_cal[i], df1_err_cal[j]))

    df1_err_cal = df1_err_cal.drop(L2, axis=1)

    return df1_err_cal