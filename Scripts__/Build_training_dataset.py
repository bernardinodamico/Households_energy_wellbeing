from DataFusion import DataFusion
import time
import datetime





'''
Script to process all the raw datasources, so to build the dataset for training the
model parameters of the Causal Bayesian Network.
'''
start_time = time.time()
dp = DataFusion(subset_only=False, how_many=None)

dp.filter_for_main_fuel_type()
dp.filter_for_method_of_payment()
dp.filter_for_income()

dp.aggregate_wall_type()

dp.fill_in_gas_cnsmp_data() 
dp.fill_in_ind_temp_data()
dp.fill_in_gas_cost_data()
dp.fill_in_energy_burden_data()
dp.fill_in_W_binary(fuel_poverty_treshold=0.1)

dp.filter_for_en_burden()

dp.ds_obsrv_vars.drop('Mainfueltype', axis=1, inplace=True)
dp.ds_obsrv_vars.drop('gasmop', axis=1, inplace=True)
dp.ds_obsrv_vars.drop('litecost', axis=1, inplace=True)

dp.rearrange_cols()

'''
NOTE method below to be changed...see inside
'''
dp.discretise() 

# save processed datasets to csv files
dp.ds_obsrv_vars.to_csv(path_or_buf="DATA/processed_dataset.csv", index=False)
dp.discrete_ds_obsrv_vars.to_csv(path_or_buf="DATA/discretised_processed_dataset.csv", index=False)
end_time = time.time()

print("Processing time (h:m:s) ", str(datetime.timedelta(seconds = round(end_time - start_time, 0))))






