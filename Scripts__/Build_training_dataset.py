from DataFusion import DataProcessing
import time

'''
Script to process all the raw datasources, so to build the dataset for training the
model parameters of the Causal Bayesian Network.
'''

dp = DataProcessing()
dp.initialise_dset_obsrv_vars(first100rows_only=True)
dp.filter_for_main_fuel_type()
dp.filter_for_income()
dp.aggregate_wall_type()

start_time = time.time()
dp.fill_in_gas_cnsmp_data() # this takes approx. 30 min to run on the full dataset
end_time = time.time()
print("Elapsed time (s): ", end_time - start_time)


dp.ds_obsrv_vars.to_csv(path_or_buf="DATA/processed_dataset.csv", index=False)




