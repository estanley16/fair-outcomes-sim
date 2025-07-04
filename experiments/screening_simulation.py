import pandas as pd
import numpy as np


class ScreeningSimulation:
    def __init__(self, total_n, frac_priv, years, prob_dict,
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
                 treatmentSuccess, cancer_prob_with_age, truncated_exponential):

        self.total_n = total_n
        self.frac_priv = frac_priv
        self.years = years
        self.prob_dict = prob_dict
        self.SEI_avg_priv_t0 = SEI_avg_priv_t0
        self.SEI_avg_unpriv_t0 = SEI_avg_unpriv_t0
        self.mortality_risk_factor_min = mortality_risk_factor_min
        self.mortality_risk_factor_max = mortality_risk_factor_max
        self.age_min = age_min
        self.age_max = age_max
        self.max_age_influx = max_age_influx

        self.screening_fxn_loc = screening_fxn_loc
        self.screening_fxn_scale = screening_fxn_scale
        self.fp_screen_reduction_frac = fp_screen_reduction_frac

        self.AI_loc_priv = AI_loc_priv
        self.AI_loc_unpriv = AI_loc_unpriv
        self.AI_scale_priv = AI_scale_priv
        self.AI_scale_unpriv = AI_scale_unpriv
        self.p_FP_AI_priv = p_FP_AI_priv
        self.p_FP_AI_unpriv = p_FP_AI_unpriv

        self.radiologist_loc = radiologist_loc
        self.radiologist_scale = radiologist_scale
        self.p_FP_radiologist = p_FP_radiologist
        self.radiologist_alpha = radiologist_alpha

        self.success_rate_biopsy = success_rate_biopsy

        self.followup_fxn_loc = followup_fxn_loc
        self.followup_fxn_scale = followup_fxn_scale

        self.delay_fxn_loc = delay_fxn_loc
        self.delay_fxn_scale = delay_fxn_scale
        self.yearly_mortality_risk_increase = yearly_mortality_risk_increase

        self.t1_risk_reduction = t1_risk_reduction
        self.t1_cost_priv = t1_cost_priv
        self.t1_cost_unpriv = t1_cost_unpriv

        self.second_treatment_fxn_loc = second_treatment_fxn_loc
        self.second_treatment_fxn_scale = second_treatment_fxn_scale
        self.t2_risk_reduction = t2_risk_reduction
        self.t2_cost_priv = t2_cost_priv
        self.t2_cost_unpriv = t2_cost_unpriv

        self.mortality_risk_factor_thresh = mortality_risk_factor_thresh

        # functions
        self.addToPool = addToPool
        self.screening = screening
        self.AI = AI
        self.radiologist = radiologist
        self.diagnose = diagnose
        self.calculate_screening_performance = calculate_screening_performance
        self.changeStatus = changeStatus
        self.test_followup = test_followup
        self.treatment_delay = treatment_delay
        self.first_treatment = first_treatment
        self.second_treatment = second_treatment
        self.second_treatment_selection = second_treatment_selection
        self.treatmentSuccess = treatmentSuccess
        self.cancer_prob_with_age = cancer_prob_with_age
        self.truncated_exponential = truncated_exponential

    def run(self):
        n_priv = int(self.total_n * self.frac_priv)
        n_unpriv = int(self.total_n - n_priv)

        df_priv = self.addToPool(n_priv, np.random.uniform(self.age_min, self.age_max, n_priv), self.prob_dict,
                                 self.SEI_avg_priv_t0, self.mortality_risk_factor_min, self.mortality_risk_factor_max, 'priv')
        df_unpriv = self.addToPool(n_unpriv, np.random.uniform(self.age_min, self.age_max, n_unpriv), self.prob_dict,
                                   self.SEI_avg_unpriv_t0, self.mortality_risk_factor_min, self.mortality_risk_factor_max, 'unpriv')

        self.df = pd.concat([df_priv, df_unpriv], ignore_index=True)
        self.df['ID'] = np.arange(0, self.total_n, 1)
        self.df['yrs_in_pool'] = 0
        self.df['last_screen'] = np.nan
        self.df['state'] = 'pool'

        self.stats = []
        for year in range(self.years):
            # print(f'year: {year}')
            if year % 10 == 0: 
                print(f'{year}')
            self._simulate_year(year)
        return pd.DataFrame(self.stats)

    def _simulate_year(self, year):
        # record number of individuals with and without BC
        n_BC_priv = len(self.df[(self.df['group']=='priv') & (self.df['BC']==1)])
        n_BC_unpriv = len(self.df[(self.df['group']=='unpriv') & (self.df['BC']==1)])

        n_healthy_priv = len(self.df[(self.df['group']=='priv') & (self.df['BC']==0)])
        n_healthy_unpriv = len(self.df[(self.df['group']=='unpriv') & (self.df['BC']==0)])

        n_priv_total = len(self.df[self.df['group']=='priv'])
        n_unpriv_total = len(self.df[self.df['group']=='unpriv'])

        # ------------------ 1. Screening/Computer-aided diagnosis ------------------ #

        # select individuals to be screened
        screened_idx = self.screening(self.df, self.screening_fxn_loc, self.screening_fxn_scale, self.fp_screen_reduction_frac)
        self.df.loc[screened_idx, 'state'] = 'CAD'
        screened_IDs = self.df.loc[self.df['state'] == 'CAD', 'ID']

        # determine results of AI CAD for all individuals being screened
        df_cad = self.df.loc[self.df['state'] == 'CAD'].copy()

        # record number from each group getting screened
        n_s_priv = len(df_cad[df_cad['group'] == 'priv'])
        n_s_unpriv = len(df_cad[df_cad['group'] == 'unpriv'])

        screen_ratio = n_s_unpriv/n_s_priv if n_s_priv>0 else np.nan
        screen_frac = (n_s_unpriv+n_s_priv)/(n_priv_total+n_unpriv_total)
        # print(f'screen ratio (unpriv/priv): {screen_ratio}')
        # print(f'screened overall: {screen_frac}\n')


        # get AI-assisted diagnosis
        AI_outputs = self.AI(df_cad, self.AI_loc_priv, self.AI_loc_unpriv, self.AI_scale_priv, self.AI_scale_unpriv, self.p_FP_AI_priv, self.p_FP_AI_unpriv)
        diagnosis = self.radiologist(df_cad, self.radiologist_loc, self.radiologist_scale, self.p_FP_radiologist, AI_outputs, self.radiologist_alpha)
        df_cad = self.diagnose(df_cad, diagnosis)

        # screening performance 
        TPR, FPR, FNR, TNR, _ = self.calculate_screening_performance(df_cad)
        TPR_priv, FPR_priv, FNR_priv, TNR_priv, _ = self.calculate_screening_performance(df_cad.loc[df_cad['group']=='priv'])
        TPR_unpriv, FPR_unpriv, FNR_unpriv, TNR_unpriv, _ = self.calculate_screening_performance(df_cad.loc[df_cad['group']=='unpriv'])

        # update state based on screening results
        df_cad.loc[df_cad['screened'].isin(['TP', 'FP']), 'state'] = 'screened_pos'
        df_cad.loc[df_cad['screened'].isin(['TN', 'FN']), 'state'] = 'screened_neg'

        # update original dataframe by adding the screened individuals back
        self.df = pd.concat([self.df[~self.df['ID'].isin(screened_IDs)], df_cad])
    
        # 1.1 Individuals screened negative are sent back to pool
        self.df = self.changeStatus(self.df, 'state', 'screened_neg', 'state', 'pool')


        # 1.2 Determine who among those screened positive are lost to follow-up
        df_all_pos = self.df.loc[self.df['state']=='screened_pos'].copy() #all positive screened individuals
        all_pos_IDs = df_all_pos['ID']
            
        lost_followup_idx = self.test_followup(df_all_pos, self.followup_fxn_loc, self.followup_fxn_scale)
        df_all_pos.loc[lost_followup_idx, 'state'] = 'lost_followup'
        
        # record number from each group lost to follow-up
        n_lost_followup_priv = len(df_all_pos.loc[(df_all_pos['state']=='lost_followup')&(df_all_pos['group']=='priv')])
        n_lost_followup_unpriv = len(df_all_pos.loc[(df_all_pos['state']=='lost_followup')&(df_all_pos['group']=='unpriv')])
        n_all_pos_priv = len(df_all_pos.loc[df_all_pos['group']=='priv'])
        n_all_pos_unpriv = len(df_all_pos.loc[df_all_pos['group']=='unpriv'])

        no_followup_ratio = n_lost_followup_unpriv/n_lost_followup_priv if n_lost_followup_priv>0 else np.nan
        no_followup_frac = (n_lost_followup_priv+n_lost_followup_unpriv)/len(df_all_pos)
        # print(f'follow-up ratio (unpriv/priv): {no_followup_ratio}')
        # print(f'lost to follow up overall: {no_followup_frac}\n')

        # update original dataframe by adding all screened pos individuals back
        self.df = pd.concat([self.df[~self.df['ID'].isin(all_pos_IDs)], df_all_pos])
        
        # send individuals lost to follow up back to pool
        self.df = self.changeStatus(self.df, 'state', 'lost_followup', 'state', 'pool')

        # update variable corresponding to the individual's most recent screening result (only for those who got diagnostic confirmation)
        self.df['last_screen'] = pd.Series(index=self.df.index, dtype='object')
        self.df.loc[self.df['state']=='screened_pos', 'last_screen'] = self.df['screened']


        # 1.3 Diagnostic confirmation step - FP individuals sent back to pool based on biopsy test efficacy
        df_FP = self.df.loc[self.df['screened']=='FP'].copy()
        self.df.loc[df_FP.sample(frac=self.success_rate_biopsy).index, 'state'] = 'pool'


        # ------------------ 2. Treatment ------------------ #

        # 2.1 Treatment delays
        df_pos = self.df.loc[self.df['state']=='screened_pos'].copy() #all true positive screened individuals

        delayed_idx = self.treatment_delay(df_pos, self.delay_fxn_loc, self.delay_fxn_scale)
        df_pos.loc[delayed_idx, 'state'] = 'delay'

        # record number from each group experiencing delays
        n_d_priv =len(df_pos.loc[(df_pos['group']=='priv')&(df_pos['state']=='delay')])
        n_d_unpriv = len(df_pos.loc[(df_pos['group']=='unpriv')&(df_pos['state']=='delay')])

        n_pos_priv = len(df_pos.loc[df_pos['group']=='priv'])
        n_pos_unpriv = len(df_pos.loc[df_pos['group']=='unpriv'])

        delay_ratio = n_d_unpriv/n_d_priv if n_d_priv>0 else np.nan
        delay_frac = (n_d_priv+n_d_unpriv)/len(df_pos)
        # print(f'delay ratio (unpriv/priv): {delay_ratio}')
        # print(f'delay overall: {delay_frac}\n')

        # increase disease progression for delayed BC+ individuals by 1/2 year factor
        df_pos.loc[df_pos.loc[(df_pos['state']=='delay') & (df_pos['BC']==1)].index, 'mortality_risk_factor'] += 0.5*self.yearly_mortality_risk_increase

        # 2.2 Apply first treatment to all individuals screened as positive
        df_pos.loc[:,'mortality_risk_factor'] = df_pos.apply(self.first_treatment, args=(self.t1_risk_reduction,), axis=1)

        # socioeconomic burden of treatment 1
        df_pos.loc[df_pos.loc[df_pos['group']=='priv'].index, 'SEI'] -= self.t1_cost_priv
        df_pos.loc[df_pos.loc[df_pos['group']=='unpriv'].index, 'SEI'] -= self.t1_cost_unpriv

        # 2.3 secondary (adjuvant) treatment is only undergone by a portion of the population
        second_treatment_idx = self.second_treatment_selection(df_pos, self.second_treatment_fxn_loc, self.second_treatment_fxn_scale)
        df_pos.loc[second_treatment_idx, 'state'] = 't2'

        # record number from each group getting adjuvant treatment
        n_t_priv =len(df_pos.loc[(df_pos['group']=='priv')&(df_pos['state']=='t2')])
        n_t_unpriv = len(df_pos.loc[(df_pos['group']=='unpriv')&(df_pos['state']=='t2')])

        t2_ratio = n_t_unpriv/n_t_priv if n_t_priv>0 else np.nan
        t2_frac = (n_t_priv+n_t_unpriv)/len(df_pos)
        # print(f't2 ratio (unpriv/priv): {t2_ratio}')
        # print(f't2 overall: {t2_frac}\n')

        # apply second treatment
        df_pos.loc[df_pos.loc[df_pos['state']=='t2'].index, 'mortality_risk_factor'] = df_pos.apply(self.second_treatment, args=(self.t2_risk_reduction,), axis=1)

        # socioeconomic burden of treatment 2
        df_pos.loc[df_pos.loc[(df_pos['group']=='priv')&(df_pos['state']=='t2')].index, 'SEI'] -= self.t2_cost_priv
        df_pos.loc[df_pos.loc[(df_pos['group']=='unpriv')&(df_pos['state']=='t2')].index, 'SEI'] -= self.t2_cost_unpriv


        # 2.4 change state based on treatment success 
        df_pos['state'] = df_pos.apply(self.treatmentSuccess, args=(self.mortality_risk_factor_thresh,), axis=1)
        #change BC status to healthy if cancer free
        df_pos = self.changeStatus(df_pos, 'state', 'cancer_free', 'BC', 0)

        # update original dataframe by adding the screened_pos individuals back
        screened_pos_IDs = df_pos['ID']
        self.df = pd.concat([self.df[~self.df['ID'].isin(screened_pos_IDs)], df_pos])

        # Add treated individuals back to pool
        self.df = self.changeStatus(self.df, 'state', 'cancer_free', 'state', 'pool')

        # ------------------ 3. End of screening cycle------------------ #

        # 3.1: Compute BC mortality in pool
        self.df.loc[self.df.loc[self.df['BC']==1].index, 'mortality_risk_factor'] += self.yearly_mortality_risk_increase #add year's worth of disease progression
        self.df.loc[self.df.loc[self.df['mortality_risk_factor']>=self.mortality_risk_factor_thresh].index, 'state'] = 'died' #change status of those who died that year

        # 3.2 Track stats for the year
        df_priv = self.df.loc[self.df['group']=='priv']
        df_unpriv = self.df.loc[self.df['group']=='unpriv']

        screen_counts_priv = df_priv.screened.value_counts(dropna=False)
        screen_counts_unpriv = df_unpriv.screened.value_counts(dropna=False)

        self.stats.append({'year': year,
                    'n_priv': len(df_priv),
                    'n_unpriv': len(df_unpriv),
                    'n_healthy_priv': n_healthy_priv,
                    'n_healthy_unpriv': n_healthy_unpriv,
                    'n_BC_priv': n_BC_priv,
                    'n_BC_unpriv': n_BC_unpriv,
                    'n_died_priv': df_priv.state.value_counts().get('died', 0),
                    'n_died_unpriv': df_unpriv.state.value_counts().get('died', 0),
                    'n_screened_priv': n_s_priv,
                    'n_screened_unpriv': n_s_unpriv,
                    'n_screened_pos_priv': n_all_pos_priv,
                    'n_screened_pos_unpriv': n_all_pos_unpriv,
                    'n_treated_priv': n_pos_priv, 
                    'n_treated_unpriv': n_pos_unpriv, 
                    'n_lost_priv': n_lost_followup_priv,
                    'n_lost_unpriv': n_lost_followup_unpriv,
                    'n_delay_priv': n_d_priv,
                    'n_delay_unpriv': n_d_unpriv,
                    'n_t2_priv': n_t_priv,
                    'n_t2_unpriv': n_t_unpriv,
                    'SEI_avg_priv': df_priv.SEI.mean(),
                    'SEI_avg_unpriv': df_unpriv.SEI.mean(),
                    'SEI_med_priv': df_priv.SEI.median(),
                    'SEI_med_unpriv': df_unpriv.SEI.median(),
                    'SEI_min_priv': df_priv.SEI.min(),
                    'SEI_min_unpriv': df_unpriv.SEI.min(),
                    'SEI_max_priv': df_priv.SEI.max(),
                    'SEI_max_unpriv': df_unpriv.SEI.max(),
                    'TP_priv': screen_counts_priv.get('TP', 0),
                    'TP_unpriv': screen_counts_unpriv.get('TP', 0),
                    'TN_priv': screen_counts_priv.get('TN', 0),
                    'TN_unpriv': screen_counts_unpriv.get('TN', 0),
                    'FP_priv': screen_counts_priv.get('FP', 0),
                    'FP_unpriv': screen_counts_unpriv.get('FP', 0),
                    'FN_priv': screen_counts_priv.get('FN', 0),
                    'FN_unpriv': screen_counts_unpriv.get('FN', 0),
                    'TPR': TPR,
                    'TNR': TNR,
                    'FPR': FPR,
                    'FNR': FNR,
                    'TPR_priv': TPR_priv,
                    'TPR_unpriv': TPR_unpriv,
                    'TNR_priv': TNR_priv,
                    'TNR_unpriv': TNR_unpriv,
                    'FPR_priv': FPR_priv,
                    'FPR_unpriv': FPR_unpriv,
                    'FNR_priv': FNR_priv,
                    'FNR_unpriv': FNR_unpriv,
                    'screen_frac': screen_frac,
                    'screen_ratio': screen_ratio, 
                    'lost_to_followup_frac': no_followup_frac,
                    'lost_to_followup_ratio': no_followup_ratio,
                    'delay_ratio': delay_ratio,
                    'delay_frac': delay_frac,
                    't2_ratio': t2_ratio, 
                    't2_frac': t2_frac,
                    'average_age_in_pool': self.df['age'].mean(),
                    })

        # 3.3 increase age, then remove dead and aged out (75+) and add new influx of individuals (ages=40-45)
        #increase ages by 1 yr
        self.df.loc[:,'age']+=1

        init_size = len(self.df) #initial popululation size

        #remove all individuals greater than 75 yrs old and those who died
        self.df = self.df[self.df['age'] <= 75]
        self.df = self.df[self.df['state'] != 'died']

        final_size = len(self.df) #popululation size after removal

        #increase yrs_in_pool
        self.df.loc[:,'yrs_in_pool']+=1

        #3.4 add new BC diagnoses to existing pool, exlcluding those who already have BC
        df_undiagnosed = self.df.loc[self.df['BC']==0].copy() #everyone in the pool currently without BC
        undiagnosed_IDs = df_undiagnosed['ID']

        #get probability mask and change the BC column where probability mask is true
        BC_mask = self.cancer_prob_with_age(self.prob_dict, df_undiagnosed['age'].to_numpy())
        df_undiagnosed.loc[BC_mask, 'BC'] = 1

        #add risk scores for individuals with BC
        df_undiagnosed['mortality_risk_factor'] = df_undiagnosed.apply(lambda x: self.truncated_exponential(1, self.mortality_risk_factor_min, self.mortality_risk_factor_max)[0] if x['BC']==1 else 0, axis=1)

        #combine back with individuals who already had BC
        self.df = pd.concat([self.df[~self.df['ID'].isin(undiagnosed_IDs)], df_undiagnosed])

        # 3.5 add new influx of individuals (ages=40-45)
        total_influx = init_size-final_size

        influx_n_priv = int(total_influx/2)
        influx_n_unpriv = int(total_influx - influx_n_priv)

        influx_priv = self.addToPool(influx_n_priv, np.random.uniform(self.age_min, self.max_age_influx, influx_n_priv), self.prob_dict, self.SEI_avg_priv_t0, self.mortality_risk_factor_min, self.mortality_risk_factor_max, 'priv')
        influx_unpriv = self.addToPool(influx_n_unpriv, np.random.uniform(self.age_min, self.max_age_influx, influx_n_unpriv), self.prob_dict, self.SEI_avg_unpriv_t0, self.mortality_risk_factor_min, self.mortality_risk_factor_max, 'unpriv')

        influx_df = pd.concat([influx_priv, influx_unpriv], ignore_index=True)
        influx_df['ID'] = np.arange(self.df.ID.max()+1, self.df.ID.max()+1+len(influx_df), 1)
        influx_df['yrs_in_pool'] = 0
        influx_df['state'] = 'pool'

        self.df = pd.concat([self.df, influx_df], ignore_index=True)

        # 3.6 reset screened column
        self.df.loc[:, 'screened'] = np.nan

        pass
