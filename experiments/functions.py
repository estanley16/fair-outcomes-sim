#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 11:31:00 2024

@author: emma
"""
import numpy as np
import pandas as pd
from scipy import special 


def cancer_prob_with_age(prob_dict, ages):
    '''
    generates a mask corresponding to individuals with BC based on age of the individual
    
    prob_dict: dictionary where keys = lower bound of age range, values = probability of getting BC
    ages: numpy array of ages

    returns mask where True corresponds to individuals with BC
    '''
    # get sorted age bins
    sorted_ages = sorted(prob_dict.keys())

    # function to get probability based on age
    def get_probability(age):
        for i in range(len(sorted_ages) - 1):
            lower_bound = sorted_ages[i]
            upper_bound = sorted_ages[i + 1]

            if lower_bound <= age < upper_bound:
                return prob_dict[lower_bound]  # return the probability of the lower bound

        # if the age is above the highest range, return the highest probability
        if age >= sorted_ages[-1]:
            return prob_dict[sorted_ages[-1]]

    # get probability for each individual based on age
    vectorized_get_probability = np.vectorize(get_probability)
    probabilities = vectorized_get_probability(ages)

    # generate a mask using binomial distribution (bernoulli trials - probabilistic sampling)
    idx_BC = np.random.binomial(n=1, p=probabilities, size=len(ages)).astype(bool) #mask corresponding to indices of people who have BC

    return idx_BC
    
def addToPool(n, ages, prob_dict, SEI_avg, mortality_risk_factor_min, mortality_risk_factor_max, group_name):
    '''
    generate dataframe of individuals of a particular group (privileged or unprivileged)

    n: number of individuals
    ages: array of ages
    prob_dict: dict defining probability of BC for a given age range
    SEI_avg: average socioeconomic index score
    mortality_risk_factor_min: lowest value of mortality risk factor for individuals with BC
    mortality_risk_factor_max: highest value of mortality risk factor for individuals with BC
    group name: 'priv' or 'unpriv'
    '''

    # get socioeconomic index distribution for group
    SEI = np.random.normal(loc=SEI_avg, scale=0.5, size=n)

    # make df
    columns = ['ID', 'age', 'group', 'SEI', 'BC', 'mortality_risk_factor', 'state', 'yrs_in_pool']
    df = pd.DataFrame(columns=columns)
    df['age'] = ages
    df['group'] = group_name
    df['SEI'] = SEI
    df['BC'] = np.zeros(n, dtype=int)

    # get probability mask and change the BC column where probability mask is true
    BC_mask = cancer_prob_with_age(prob_dict, ages)
    df.loc[BC_mask, 'BC'] = 1

    # add risk scores for individuals with BC
    df['mortality_risk_factor'] = df.apply(lambda x: truncated_exponential(1, mortality_risk_factor_min, mortality_risk_factor_max)[0] if x['BC']==1 else 0, axis=1)
    return df

def changeStatus(df, conditionalCol, conditionalVal, targetCol, targetVal):
    '''
    change entry in target column to target value based on value of another (conditional) column

    conditionalCol: column name for the conditional statement
    conditionalVal: value of conditionalCol that triggers a change in the value of the target column
    targetCol: column name for the variable to change
    targetVal: value to change targetCol to

    e.g.  for changing breast cancer status to 0 if patient is cancer free:
    conditionalCol = state
    conditionalVal = cancer_free
    target_col = BC
    target_val = 0
    '''
    df.loc[df.loc[df[conditionalCol]==conditionalVal].index, targetCol] = targetVal
    return df

def screening(df, loc, scale, fp_screen_reduction_frac): 
    '''
    determine who in the pool gets screened w logistic fxn --> p(screened|SEI)
    df: dataframe with everyone in pool
    loc: location param of logistic probability function
    scale: scale param of logistic probability function
    fp_screen_reduction_frac: reduction in screening probability due to previous false positive
    returns mask corresponding to indices of people who got screened
    '''
    probabilities = special.expit(df['SEI']*scale - loc)

    # if last screening result was false positive, reduce screening probability
    FP_mask = df['last_screen'] == 'FP'
    probabilities = np.where(FP_mask, 
                            np.maximum(0, probabilities - fp_screen_reduction_frac),
                            probabilities)

    idx_screened = np.random.binomial(n=1, p=probabilities, size=len(df)).astype(bool) #mask corresponding to indices of people who got screened

    return idx_screened
    
def AI(df, loc_priv, loc_unpriv, scale_priv, scale_unpriv, p_FP_priv, p_FP_unpriv):
    '''
    compute AI model outputs for people being screened --> f = p(BC|risk score, group)
    df: dataframe with everyone in screening stage 
    loc_priv, loc_unpriv: location param of logistic probability function, defines group-wise performance
    scale: scale param of logistic probability function (same for both groups)
    p_FP_priv, p_FP_unpriv: probability of a false positive, defines group-wise performance
    returns AI model output for everyone in df (1 = breast cancer, 0 = healthy)
    '''
    
    # create arrays of loc and scale values based on group
    loc = np.where(df['group'] == 'priv', loc_priv, loc_unpriv)
    scale = np.where(df['group'] == 'priv', scale_priv, scale_unpriv)
    p_FP = np.where(df['group'] == 'priv', p_FP_priv, p_FP_unpriv) 
    
    f = np.where(df['mortality_risk_factor'] == 0,
                  p_FP,
                  scale * df['mortality_risk_factor'] + loc)
    
    return f

def radiologist(df, loc, scale, p_FP, f, alpha): 
    '''
    computes radiologist's diagnosis for people being screened --> dependent on their own decision function and on AI model
    radiologist's decision function: g = p(BC|risk score) 
    radiologist's final diagnosis, considering AI output: h = alpha(g) + (1-alpha)(f)
    
    df: dataframe with everyone in screening stage 
    loc: location param of logistic probability function for radiologist's decision function
    scale: scale param of logistic probability function for radiologist's decision function 
    p_FP: probability of a false positive for radiologist's decision function 
    f: AI model output 
    alpha: parameter defining radiologist confidence vs. overreliance on model 
        (when alpha=1, only radiologist decision matters, when alpha = 0, only AI decision matters) 
    returns diagnosis for everyone in df (1 = breast cancer, 0 = healthy)
    '''

    #first, compute clincian's own decision function 
    g = np.where(df['mortality_risk_factor'] == 0,
                  p_FP,
                  scale * df['mortality_risk_factor'] + loc) #clincian's probabilities of BC
    
    #then, compute AI + radiologist combined probability of BC 
    h = alpha*f + (1-alpha)*g 

    #finally, Bernoulli trial to get radiologist's final diagnostic decision
    diagnosis = np.random.binomial(n=1, p=h, size=len(df))

    return diagnosis

def diagnose(df, diagnosis):
    # true positives: actually has cancer (BC=1) and diagnosed positive (diagnosis=1)
    df.loc[(df['BC'] == 1) & (diagnosis == 1), 'screened'] = 'TP'
    
    # false negatives: actually has cancer (BC=1) but diagnosed negative (diagnosis=0)
    df.loc[(df['BC'] == 1) & (diagnosis == 0), 'screened'] = 'FN'
    
    # false positives: actually healthy (BC=0) but diagnosed positive (diagnosis=1)
    df.loc[(df['BC'] == 0) & (diagnosis == 1), 'screened'] = 'FP'
    
    # true negatives: actually healthy (BC=0) and diagnosed negative (diagnosis=0) 
    df.loc[(df['BC'] == 0) & (diagnosis == 0), 'screened'] = 'TN'

    return df

def test_followup(df, loc, scale): 
    '''
    determine who among those screened positive do not proceed to diagnostic confirmation w logistic fxn --> p(diagnostic confirmation|SEI)
    df: dataframe with everyone screened positive
    loc: location param of logistic probability function
    scale: scale param of logistic probability function
    returns mask corresponding to indices of people who did not proceed to diagnostic confirmation (returned to pool)
    '''
    probabilities_followup = special.expit(df['SEI']*scale - loc)

    probabilities_no_followup = 1-probabilities_followup 

    idx_no_followup = np.random.binomial(n=1, p=probabilities_no_followup, size=len(df)).astype(bool) #mask corresponding to indices of people who did not proceed to follow up

    return idx_no_followup

def calculate_screening_performance(df):
    counts = df['screened'].value_counts()
    TN = counts['TN']
    TP = counts['TP']
    FN = counts['FN']
    FP = counts['FP']

    TPR = TP/(TP+FN)*100
    FPR  = FP/(FP+TN)*100
    FNR = FN/(TP+FN)*100
    TNR = TN/(TN+FP)*100
    precision = TP/(TP+FP)*100

    # print(f'True positive rate: {TPR}')
    # print(f'False positive rate: {FPR}')
    # print(f'False negative rate: {FNR}')
    # print(f'True negative rate: {TNR}')
    # print(f'Precision: {precision}')

    return TPR, FPR, FNR, TNR, precision
    
def treatment_delay(df, loc, scale): 
    '''
    determine who experiences treatment delays w logistic fxn --> p(treatment delay|SEI)
    df: dataframe with everyone who was screened positive
    loc: location param of logistic probability function
    scale: scale param of logistic probability function
    returns mask corresponding to indices of people who experienced treatment delays
    '''
    probabilities_no_delay = special.expit(df['SEI']*scale - loc)

    probabilities_delay = 1-probabilities_no_delay 
    
    idx_delay = np.random.binomial(n=1, p=probabilities_delay, size=len(df)).astype(bool) #mask corresponding to indices of people who experience delays

    return idx_delay


def first_treatment(indiv, t1_risk_reduction):
    '''
    function that simulates the first stage of breast cancer treatment that both groups undergo (i.e. surgery)
    changes the severity/mortality risk by a value that represents treatment effectiveness in preventing mortality

    indiv: row of dataframe for an individual
    '''
    if indiv['BC'] == 0: #if they were a false positive
        return
    else:
        return indiv['mortality_risk_factor']-t1_risk_reduction

def second_treatment_selection(df, loc, scale): 
    '''
    determine who in the pool undergoes secondary treatment w logistic fxn --> p(secondary treatment|SEI)
    df: dataframe with everyone who was screened positive 
    loc: location param of logistic probability function
    scale: scale param of logistic probability function
    returns mask corresponding to indices of people who got secondary treatment
    '''
    probabilities = special.expit(df['SEI']*scale - loc)

    idx_t2 = np.random.binomial(n=1, p=probabilities, size=len(df)).astype(bool) #mask corresponding to indices of people who experience delays

    return idx_t2
    
def second_treatment(indiv, t2_risk_reduction):
    '''
    function that simulates the second stage of breast cancer treatment that only a fraction of the each group undergoes (i.e. adjuvant)
    changes the severity/mortality risk by a value that represents treatment effectiveness in preventing mortality

    indiv: row of dataframe for an individual
    '''
    if indiv['BC'] == 0: #if they were a false positive
        return
    else:
        return indiv['mortality_risk_factor']-t2_risk_reduction
        
def treatmentSuccess(indiv, thresh):
    '''
    function that determines likelihood of treatment success given risk score
    updates the 'state' variable to cancer_free or died

    indiv: row of dataframe for an individual
    thresh: threshold above which the patient dies
    '''
    if indiv['BC'] == 0: #if they were a false positive
        return 'cancer_free'

    if indiv['mortality_risk_factor'] <= thresh:
        return 'cancer_free'

    else: return 'died'

def fit_linear(x1, y1, x2, y2):
    m = (y2-y1)/(x2-x1)
    b = y2 - m*x2
    return m, b

def truncated_exponential(lambda_, a, b, size=1):
    """
    Sample from a truncated exponential distribution between [a, b].

    Parameters:
    lambda_ : float
        The rate parameter of the exponential distribution.
    a : float
        The lower bound of truncation.
    b : float
        The upper bound of truncation.
    size : int
        The number of samples to draw.

    Returns:
    samples : ndarray
        Samples drawn from the truncated exponential distribution.
    """

    # Inverse CDF (quantile function) of the exponential distribution
    def inverse_cdf(u, lambda_):
        return -np.log(1 - u) / lambda_

    # Calculate the CDF values at the boundaries a and b
    cdf_a = 1 - np.exp(-lambda_ * a)
    cdf_b = 1 - np.exp(-lambda_ * b)

    # Sample uniform values in the adjusted [cdf_a, cdf_b] range
    u = np.random.uniform(cdf_a, cdf_b, size=size)

    # Apply the inverse CDF to transform the uniform samples to exponential samples
    samples = inverse_cdf(u, lambda_)

    return samples

