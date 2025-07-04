#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 11:29:25 2024

@author: emma
"""
import numpy as np
from functions import fit_linear

#simulation parameters
years = 51 #number of years to simulate
total_n = 500000 #total population size
frac_priv = 0.5 #fraction of privileged group
num_sims = 100 #number of simulations to run 

#baseline probability function parameters 
screening_fxn_loc = -6.25
screening_fxn_scale = 5
followup_fxn_loc = -2
followup_fxn_scale = 0.77
fp_screen_reduction_frac = 0.10
delay_fxn_loc = -1.2
delay_fxn_scale = 0.95
second_treatment_fxn_loc = -1.8
second_treatment_fxn_scale = 0.45

#radiologist decision functions - approx. probability of positive breast cancer diagnosis as a function of risk score
risk_scores = np.array([0, 1, 4])
radiologist_probabilities = np.array([0.1, 0.65, 0.90])
radiologist_alpha = 0.5

# we only fit the linear function to risk scores between 1 and 5 (individuals with BC=1); p(BC) @ R=0 will be represented in piecewise fxn
radiologist_scale, radiologist_loc = fit_linear(risk_scores[1], radiologist_probabilities[1], risk_scores[2], radiologist_probabilities[2])

#other part of piecewise: probability of false positive 
p_FP_radiologist = radiologist_probabilities[0]

SEI_avg_priv_t0 = 1 #mean socioeconomic index for privileged group
SEI_avg_unpriv_t0 = -1 #mean socioeconomic index for unprivileged group
mortality_risk_factor_min = 1 #lowest value of mortality risk factor for individuals with BC
mortality_risk_factor_max = 4 #highest value of mortality risk factor for individuals with BC
mortality_risk_factor_thresh = 3 #threshold of mortality risk factor after which death occurs
yearly_mortality_risk_increase = 1 #factor that mortality risk increases per year
age_min = 40 #min age in pool eligible for screening
age_max = 75 #max age in pool eligble for screening

#dict defining probability of BC for a given age range
#based on SEER data - https://seer.cancer.gov/archive/csr/1975_2011/results_merged/sect_04_breast.pdf table 4.12 - age specific incidence rates 
prob_dict = {40: 1.225e-3,
          45: 1.886e-3,
          50: 2.243e-3,
          55: 2.664e-3,
          60: 3.467e-3,
          65: 4.202e-3,
          70: 4.33e-3}


success_rate_biopsy = 1 #fractional rate of how accurate the biopsy analysis is (0.99 would be 99% accurate, 1% would remain FP)

#cost for each stage of treatment
t1_cost_priv = 0.5
t1_cost_unpriv = 2*t1_cost_priv

t2_cost_priv = 0.5
t2_cost_unpriv = 2*t2_cost_priv


t1_risk_reduction = 1 #relative mortality risk reduction of undergoing first treatment
t2_risk_reduction = 0.5 #relative mortality risk reduction of undergoing second treatment

max_age_influx = 45 #max age of new individuals added to screening pool
