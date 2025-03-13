from DataFusion import DataFusion
import time

'''
Script to process all the raw datasources, so to build the dataset for training the
model parameters of the Causal Bayesian Network.
'''

dp = DataFusion()
dp.initialise_dset_obsrv_vars(first100rows_only=True)
dp.filter_for_main_fuel_type()
dp.filter_for_method_of_payment()
dp.filter_for_income()
dp.aggregate_wall_type()

start_time = time.time()
#dp.fill_in_gas_cnsmp_data() # this takes approx. 30 min to run on the full dataset
dp.fill_in_ind_temp_data()
end_time = time.time()
print("Elapsed time (s): ", end_time - start_time)

dp.ds_obsrv_vars.drop('Mainfueltype', axis=1, inplace=True)
dp.ds_obsrv_vars.drop('gasmop', axis=1, inplace=True)

'''
Saving the generated training dataset into 'DATA' folder
'''
dp.ds_obsrv_vars.to_csv(path_or_buf="DATA/processed_dataset.csv", index=False)




