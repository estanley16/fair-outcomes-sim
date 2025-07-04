#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

screening simulation - AI performance sensitivity analysis

"""


import numpy as np
import pandas as pd
import time
import concurrent.futures
from typing import List, Optional
import argparse
from pathlib import Path
import sys
from variables import *
from functions import *
from screening_simulation import ScreeningSimulation

#--------------------------- params --------------------------- #

exp_name = 'exp17G'
results_dir = '/home/emma/Documents/BC_Sim/results/'
Path(results_dir + exp_name).mkdir(parents=True, exist_ok=True)
save_dir = results_dir + exp_name + '/'

#list of AI performance scenarios to test
dFPR_list = [0, 0.05, 0.10, 0.15, 0.2, 0.25, 0.3]
dTPR_list = [0, 0.05, 0.10, 0.15, 0.2, 0.25, 0.3]

#define baseline probability of false positive for BOTH groups --> INCREASES BASED ON THIS
p_FP_AI = 0.05

#define baseline prob of TPs for risk scores 1 and 4 for BOTH groups --> DECREASES BASED ON THIS
p_TP_AI = np.array([0.75, 0.95])


# ------------------------------------------------------------------------------------ #

def run_single_simulation_performance_analysis(seed: int) -> pd.DataFrame:
    """
    single simulation with a specific random seed

    returns dataframe with simulation statistics
    """
    np.random.seed(seed)


    #initialize dataframes to collect outcome metrics - MRs, MR disparity, MR ratio, SEI disparity, change in SEI disparity
    MRunpriv_df = pd.DataFrame(index=dTPR_list,
                   columns=dFPR_list)
    MRpriv_df = pd.DataFrame(index=dTPR_list,
                   columns=dFPR_list)
    dMR_df = pd.DataFrame(index=dTPR_list,
                   columns=dFPR_list)
    MR_ratio_df = pd.DataFrame(index=dTPR_list,
                   columns=dFPR_list)
    dSEI_df = pd.DataFrame(index=dTPR_list,
                           columns=dFPR_list)
    pc_dSEI_df = pd.DataFrame(index=dTPR_list,
                           columns=dFPR_list)
    dSEI_med_df = pd.DataFrame(index=dTPR_list,
                           columns=dFPR_list)
    pc_dSEI_med_df = pd.DataFrame(index=dTPR_list,
                           columns=dFPR_list)


    #years to start and end averaging of outcome metrics
    yr_start = 1
    yr_end = years-1


    for dFPR in dFPR_list:

        p_FP_AI_priv = p_FP_AI + dFPR #FPR increases based on delta
        p_FP_AI_unpriv = p_FP_AI + dFPR #FPR increases based on delta

        for dTPR in dTPR_list:


            p_TP_AI_priv = p_TP_AI - dTPR #TPR dereases based on delta
            p_TP_AI_unpriv = p_TP_AI - dTPR #TPR dereases based on delta

            print(f'delta TPR = {p_TP_AI_priv - p_TP_AI_unpriv}')
            print(f'delta FPR = {p_FP_AI_priv - p_FP_AI_unpriv}')

            #get parameters for AI decision function
            AI_scale_priv, AI_loc_priv = fit_linear(risk_scores[1], p_TP_AI_priv[0], risk_scores[2], p_TP_AI_priv[1])
            #print(f'AI priv scale = {AI_scale_priv}, AI priv loc = {AI_loc_priv}')

            AI_scale_unpriv, AI_loc_unpriv = fit_linear(risk_scores[1], p_TP_AI_unpriv[0], risk_scores[2], p_TP_AI_unpriv[1])
            #print(f'AI unpriv scale = {AI_scale_unpriv}, AI priv loc = {AI_loc_unpriv}')

            sim = ScreeningSimulation(total_n, frac_priv, years, prob_dict,
                 SEI_avg_priv_t0, SEI_avg_unpriv_t0,
                 mortality_risk_factor_min, mortality_risk_factor_max,
                 age_min, age_max, max_age_influx,
                 screening_fxn_loc, screening_fxn_scale, fp_screen_reduction_frac,
                 AI_loc_priv, AI_loc_unpriv, AI_scale_priv, AI_scale_unpriv, p_FP_AI_priv, p_FP_AI_unpriv,
                 radiologist_loc, radiologist_scale, p_FP_radiologist, radiologist_alpha,
                 success_rate_biopsy,
                 followup_fxn_loc, followup_fxn_scale,
                 delay_fxn_loc, delay_fxn_scale,
                 yearly_mortality_risk_increase,
                 t1_risk_reduction, t1_cost_priv, t1_cost_unpriv,
                 second_treatment_fxn_loc, second_treatment_fxn_scale,
                 t2_risk_reduction, t2_cost_priv, t2_cost_unpriv,
                 mortality_risk_factor_thresh,
                 addToPool, screening, AI, radiologist, diagnose, 
                 calculate_screening_performance, changeStatus, test_followup,
                 treatment_delay, first_treatment, second_treatment_selection, second_treatment, 
                 treatmentSuccess, cancer_prob_with_age, truncated_exponential)
        
            stats_df = sim.run()
            

            #compute derived metrics
            #group mortality rates
            stats_df['MR_priv'] = stats_df['n_died_priv']/stats_df['n_priv']*1e5
            stats_df['MR_unpriv'] = stats_df['n_died_unpriv']/stats_df['n_unpriv']*1e5
            stats_df['deltaMR'] = stats_df['MR_unpriv']-stats_df['MR_priv']
            stats_df['MR_ratio'] = stats_df['MR_unpriv']/stats_df['MR_priv']

            #inter-group deltaSEI
            stats_df['inter_dSEI'] = np.absolute(stats_df['SEI_avg_priv']-stats_df['SEI_avg_unpriv'])
            stats_df['inter_med_dSEI'] = np.absolute(stats_df['SEI_med_priv']-stats_df['SEI_med_unpriv'])

            #percent change in inter-group deltaSEI
            stats_df['change_inter_dSEI'] = stats_df['inter_dSEI'].pct_change()*100
            stats_df['change_inter_med_dSEI'] = stats_df['inter_med_dSEI'].pct_change()*100


            #calculate average dMR and dSEI over time frame
            stats_df = stats_df.loc[yr_start:yr_end,:]
            avg_MRunpriv = stats_df['MR_unpriv'].mean()
            avg_MRpriv = stats_df['MR_priv'].mean()
            avg_dMR = stats_df['deltaMR'].mean()
            avg_MR_ratio = stats_df['MR_ratio'].mean()
            avg_dSEI = stats_df['inter_dSEI'].mean()
            avg_pc_dSEI = stats_df['change_inter_dSEI'].mean()
            avg_med_dSEI = stats_df['inter_med_dSEI'].mean()
            avg_pc_med_dSEI = stats_df['change_inter_med_dSEI'].mean()


            MRunpriv_df.loc[dTPR, dFPR] = avg_MRunpriv
            MRpriv_df.loc[dTPR, dFPR] = avg_MRpriv

            dMR_df.loc[dTPR, dFPR] = avg_dMR
            MR_ratio_df.loc[dTPR, dFPR] = avg_MR_ratio

            dSEI_df.loc[dTPR, dFPR] = avg_dSEI
            pc_dSEI_df.loc[dTPR, dFPR] = avg_pc_dSEI

            dSEI_med_df.loc[dTPR, dFPR] = avg_med_dSEI
            pc_dSEI_med_df.loc[dTPR, dFPR] = avg_pc_med_dSEI




    return MRunpriv_df, MRpriv_df, dMR_df, MR_ratio_df, dSEI_df, pc_dSEI_df, dSEI_med_df, pc_dSEI_med_df




def parallel_simulations_performance_analysis(
    n_sims: int = 100,
    n_cores: Optional[int] = None,
    timeout: Optional[float] = None
) -> pd.DataFrame:
    """
    Run multiple simulations in parallel using concurrent.futures

    Parameters:
    n_sims : int, number of simulations to run
    n_cores : int or None, number of CPU cores to use
    timeout : float or None, maximum time to wait for simulations

    Returns:
    Combined DataFrame with results from all simulations
    """
    import os

    # Determine number of cores
    if n_cores is None:
        n_cores = os.cpu_count() or 1

    # Generate unique seeds
    seeds = np.random.SeedSequence(42).spawn(n_sims)

    # Use ProcessPoolExecutor for CPU-bound tasks
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
        # Submit all tasks
        future_to_seed = {
            executor.submit(run_single_simulation_performance_analysis, seed.generate_state(1)[0]): seed
            for seed in seeds
        }

        MRunpriv_results_lst: List[pd.DataFrame] = []
        MRpriv_results_lst: List[pd.DataFrame] = []
        dMR_results_lst: List[pd.DataFrame] = []
        MR_ratio_results_lst: List[pd.DataFrame] = []
        dSEI_results_lst: List[pd.DataFrame] = []
        pc_dSEI_results_lst: List[pd.DataFrame] = []
        dSEI_med_results_lst: List[pd.DataFrame] = []
        pc_dSEI_med_results_lst: List[pd.DataFrame] = []

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_seed, timeout=timeout):
            seed = future_to_seed[future]

            MRunpriv, MRpriv, dMR, MR_ratio, dSEI, pc_dSEI, dSEI_med, pc_dSEI_med = future.result() #returns from run_single_simulation()

            MRunpriv['n_sim'] = int(seed.generate_state(1)[0])
            MRpriv['n_sim'] = int(seed.generate_state(1)[0])
            dMR['n_sim'] = int(seed.generate_state(1)[0])
            MR_ratio['n_sim'] = int(seed.generate_state(1)[0])
            dSEI['n_sim'] = int(seed.generate_state(1)[0])
            pc_dSEI['n_sim'] = int(seed.generate_state(1)[0])
            dSEI_med['n_sim'] = int(seed.generate_state(1)[0])
            pc_dSEI_med['n_sim'] = int(seed.generate_state(1)[0])


            MRunpriv_results_lst.append(MRunpriv)
            MRpriv_results_lst.append(MRpriv)
            dMR_results_lst.append(dMR)
            MR_ratio_results_lst.append(MR_ratio)
            dSEI_results_lst.append(dSEI)
            pc_dSEI_results_lst.append(pc_dSEI)
            dSEI_med_results_lst.append(dSEI_med)
            pc_dSEI_med_results_lst.append(pc_dSEI_med)


    # Combine results
    MRunpriv_results = pd.concat(MRunpriv_results_lst, ignore_index=False)
    MRpriv_results = pd.concat(MRpriv_results_lst, ignore_index=False)
    dMR_results = pd.concat(dMR_results_lst, ignore_index=False)
    MR_ratio_results = pd.concat(MR_ratio_results_lst, ignore_index=False)
    dSEI_results = pd.concat(dSEI_results_lst, ignore_index=False)
    pc_dSEI_results = pd.concat(pc_dSEI_results_lst, ignore_index=False)
    dSEI_med_results = pd.concat(dSEI_med_results_lst, ignore_index=False)
    pc_dSEI_med_results = pd.concat(pc_dSEI_med_results_lst, ignore_index=False)


    return MRunpriv_results, MRpriv_results, dMR_results, MR_ratio_results, dSEI_results, pc_dSEI_results, dSEI_med_results, pc_dSEI_med_results




if __name__ == '__main__':
    n_sims = num_sims
    start_time = time.time()
    MRunpriv_results, MRpriv_results, dMR_results, MR_ratio_results, dSEI_results, pc_dSEI_results, dSEI_med_results, pc_dSEI_med_results = parallel_simulations_performance_analysis(n_sims=n_sims)
    end_time = time.time()

    MRunpriv_results.to_csv(save_dir +'MRunpriv.csv')

    MRpriv_results.to_csv(save_dir +'MRpriv.csv')

    dMR_results.to_csv(save_dir +'dMR.csv')

    MR_ratio_results.to_csv(save_dir +'MR_ratio.csv')

    dSEI_results.to_csv(save_dir +'dSEI.csv')

    pc_dSEI_results.to_csv(save_dir +'pc_dSEI.csv')

    dSEI_med_results.to_csv(save_dir +'dSEI_med.csv')

    pc_dSEI_med_results.to_csv(save_dir +'pc_dSEI_med.csv')

    print(f'time for {n_sims} simulations: {end_time-start_time:.2f}s')
