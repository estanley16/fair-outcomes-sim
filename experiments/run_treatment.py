#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

screening simulation - adjuvant treatment sensitivity analysis

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

parser = argparse.ArgumentParser()
parser.add_argument('--ai_fairness', type=str, help='naive, EqOppTPR, EqOppFPR, EqOdds', required=True)
args = parser.parse_args()

if args.ai_fairness == 'naive':
    AI_priv_probabilities = np.array([0.20, 0.75,0.95]) #naive (biased)
    AI_unpriv_probabilities = np.array([0.05, 0.6, 0.80]) #naive (biased)

elif args.ai_fairness == 'EqOppTPR':
    AI_priv_probabilities = np.array([0.20, 0.675, 0.875]) #Eq Opp TPR
    AI_unpriv_probabilities = np.array([0.05, 0.675, 0.875]) #Eq Opp TPR

elif args.ai_fairness == 'EqOppFPR':
    AI_priv_probabilities = np.array([0.125, 0.75,0.95]) #Eq Opp TPR
    AI_unpriv_probabilities = np.array([0.125, 0.6, 0.80]) #Eq Opp TPR

elif args.ai_fairness == 'EqOdds':
    AI_priv_probabilities = np.array([0.125, 0.675, 0.875]) #Eq Opp TPR
    AI_unpriv_probabilities = np.array([0.125, 0.675, 0.875]) #Eq Opp TPR

else:
    sys.exit('did not get valid ai fairness input')



exp_name = 'exp17F'
results_dir = '/home/emma/Documents/BC_Sim/results/'
Path(results_dir + exp_name).mkdir(parents=True, exist_ok=True)
save_dir = results_dir + exp_name + '/'

AI_scale_priv, AI_loc_priv = fit_linear(risk_scores[1], AI_priv_probabilities[1], risk_scores[2], AI_priv_probabilities[2])
print(f'AI priv scale = {AI_scale_priv}, AI priv loc = {AI_loc_priv}')

AI_scale_unpriv, AI_loc_unpriv = fit_linear(risk_scores[1], AI_unpriv_probabilities[1], risk_scores[2], AI_unpriv_probabilities[2])
print(f'AI unpriv scale = {AI_scale_unpriv}, AI priv loc = {AI_loc_unpriv}')

p_FP_AI_priv = AI_priv_probabilities[0]
p_FP_AI_unpriv = AI_unpriv_probabilities[0]

t2_loc_arr = [-3.8, -1.9, -0.85, 0.25, 2.1]
# ------------------------------------------------------------------------------------ #

def run_single_simulation(seed: int,
                           years: int = 51,
                           total_n: int = 500000,
                           frac_priv: float = 0.5) -> pd.DataFrame:
    """
    Run a single simulation with a specific random seed

    Parameters:
    seed : int, random seed for reproducibility
    years : int, number of years to simulate
    total_n : int, total population size
    frac_priv : float, fraction of privileged group

    Returns:
    DataFrame with simulation statistics
    """
    np.random.seed(seed)

    try:

        t2_df_list = []

        for second_treatment_fxn_loc in t2_loc_arr:

            print(f't2 loc parameter: {second_treatment_fxn_loc}')

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

            stats_df.loc[:, 't2_loc'] = second_treatment_fxn_loc
            t2_df_list.append(stats_df)

        t2_df = pd.concat(t2_df_list, ignore_index=True)
        return t2_df

    except Exception as e:
        print(f"Simulation with seed {seed} failed: {e}")
        return pd.DataFrame()  # Return empty DataFrame on failure


def parallel_simulations(
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
            executor.submit(run_single_simulation, seed.generate_state(1)[0]): seed
            for seed in seeds
        }

        results: List[pd.DataFrame] = []

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_seed, timeout=timeout):
            seed = future_to_seed[future]
            try:
                result = future.result()
                if not result.empty:
                    result['n_sim'] = int(seed.generate_state(1)[0])
                    results.append(result)
            except Exception as exc:
                print(f'Simulation generated an exception: {exc}')

    # Combine results
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return pd.DataFrame()



if __name__ == '__main__':
    n_sims = num_sims
    start_time = time.time()
    results = parallel_simulations(n_sims=n_sims)
    end_time = time.time()

    results['ai_fairness']=args.ai_fairness
    results.to_csv(save_dir + args.ai_fairness +'.csv')

    print(f'time for {n_sims} simulations: {end_time-start_time:.2f}s')
