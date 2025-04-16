import pandas as pd
from DataFusion import gen_training_dataset
from Estimator import ComputeEffects
import numpy as np
pd.option_context('display.max_rows', None)



def placebo_tretment(tot_samples: int) -> None:
    '''
    The method generates a sample of training datasets where the values of the treatment variable X are randomly 
    shuffled (placebo tretment). Then it uses these datasets to estimate the corresponding ATE and saves these
    into a csv for further use.
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

        print(f'Random sample {random_seed} out of {tot_samples}. Placebo ATE = {ATE_placebo} kWh/year')

    return
    


if __name__ == "__main__":
    placebo_tretment(tot_samples=50)