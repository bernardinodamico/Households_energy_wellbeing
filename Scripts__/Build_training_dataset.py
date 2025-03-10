from DataProcessing import DataProcessing

'''
Script to process all the raw datasources, so to build the dataset for training the
model parameters of the Causal Bayesian Network.
'''

dp = DataProcessing()
dp.initialise_dset_obsrv_vars()
dp.filter_for_main_fuel_type()
dp.filter_for_income()
dp.aggregate_wall_type()


dp.ds_obsrv_vars.to_csv(path_or_buf="DATA/processed_dataset.csv", index=False)




