import pandas as pd
from DataFusion import gen_training_dataset
from Estimator import ComputeEffects
import numpy as np
import random
pd.option_context('display.max_rows', None)




def placebo_treatment_test(tot_samples: int) -> None:
    '''
    The method generates a sample of training datasets where the values of the treatment variable X are randomly 
    shuffled (placebo tretment). Then it uses these datasets to estimate the corresponding ATE and saves these
    into a csv for further statistical analysis.
    '''
    #set bin number for real-valued variables
    Y0bn = 13
    Wbn = 13 
    V1bn = 13
    V7bn = 13
    Laplace_sm = 0.001

    discretised_dtset = gen_training_dataset(Y_0_bins_num=Y0bn, W_bins_num=Wbn, V_1_bins_num=V1bn, V_7_bins_num=V7bn)
    shuffled_discretised_dtset = discretised_dtset.copy(deep=True)

    plecebo_treatmt_reslt = pd.DataFrame({'Random_seed': pd.Series(dtype='int'), 
                                          'ATE_placebo': pd.Series(dtype='float')})

    for random_seed in range(1, tot_samples):
        np.random.seed(random_seed)
        shuffled_discretised_dtset['X'] = np.random.permutation(discretised_dtset['X'].values)

        ce = ComputeEffects()
        _, _, exp_Y0_given_doXx_1G, exp_Y0_given_doXx_2G = ce.compute_ATE(Y_0_bins_num=Y0bn, 
                                                                          W_bins_num=Wbn, 
                                                                          V_1_bins_num=V1bn, 
                                                                          V_7_bins_num=V7bn, 
                                                                          Laplace_sm=Laplace_sm, 
                                                                          dd=shuffled_discretised_dtset)
        
        ATE_placebo = round(exp_Y0_given_doXx_2G - exp_Y0_given_doXx_1G, 3)
        new_row = {'Random_seed': random_seed, 'ATE_placebo': ATE_placebo}
        plecebo_treatmt_reslt = pd.concat([plecebo_treatmt_reslt, pd.DataFrame([new_row])], ignore_index=True)
        plecebo_treatmt_reslt.to_csv(path_or_buf="DATA/REFUTATION_TEST_RESULTS/placebo_treatment_results.csv", index=False)

        print(f'Random seed {random_seed} out of {tot_samples}. Placebo ATE = {ATE_placebo} kWh/year')

    return
    

def data_subsample_test(tot_samples: int) -> None:
    '''
    The method generates a sample of training datasets where each dataset is a random subsample of the training dataset. 
    Then it uses these random subsample datasets to estimate the corresponding ATE and saves these
    into a csv for further statistical analysis.
    '''
    #set bin number for real-valued variables
    Y0bn = 13
    Wbn = 13 
    V1bn = 13
    V7bn = 13
    Laplace_sm = 0.001

    discretised_dtset = gen_training_dataset(Y_0_bins_num=Y0bn, W_bins_num=Wbn, V_1_bins_num=V1bn, V_7_bins_num=V7bn)

    data_subsample_reslt = pd.DataFrame({'Random_seed': pd.Series(dtype='int'), 
                                          'ATE_subsample': pd.Series(dtype='float')})

    subsample_size = 0.4 # percentage of the original dataset
    for random_seed in range(1, tot_samples):
        subsample_discretised_dtset = discretised_dtset.sample(frac=subsample_size, random_state=random_seed)  

        ce = ComputeEffects()
        _, _, exp_Y0_given_doXx_1G, exp_Y0_given_doXx_2G = ce.compute_ATE(Y_0_bins_num=Y0bn, 
                                                                          W_bins_num=Wbn, 
                                                                          V_1_bins_num=V1bn, 
                                                                          V_7_bins_num=V7bn, 
                                                                          Laplace_sm=Laplace_sm, 
                                                                          dd=subsample_discretised_dtset)
        
        ATE_subsample = round(exp_Y0_given_doXx_2G - exp_Y0_given_doXx_1G, 3)
        new_row = {'Random_seed': random_seed, 'ATE_subsample': ATE_subsample}
        data_subsample_reslt = pd.concat([data_subsample_reslt, pd.DataFrame([new_row])], ignore_index=True)
        data_subsample_reslt.to_csv(path_or_buf=f"DATA/REFUTATION_TEST_RESULTS/data_subsample_results_{int(subsample_size*100)}percent.csv", index=False)

        print(f'Random seed {random_seed} out of {tot_samples}. Data subsample ATE = {ATE_subsample} kWh/year')



    return




if __name__ == "__main__":
    #placebo_tretment(tot_samples=5000)
    data_subsample_test(tot_samples=5000)