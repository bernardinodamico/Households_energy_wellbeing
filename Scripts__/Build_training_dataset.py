from DataFusion import DataFusion
import time
import datetime

'''
Script to process all the raw datasources, so to build the dataset for training the
model parameters of the Causal Bayesian Network.
'''
start_time = time.time()
dp = DataFusion()
dp.initialise_dset_obsrv_vars(subset_only=True, how_many=1000)
dp.filter_for_main_fuel_type()
dp.filter_for_method_of_payment()
dp.filter_for_income()
dp.aggregate_wall_type()

dp.fill_in_gas_cnsmp_data() 
dp.fill_in_ind_temp_data()
dp.fill_in_gas_cost_data()
dp.fill_in_energy_burden_data()
dp.ds_obsrv_vars.drop('Mainfueltype', axis=1, inplace=True)
dp.ds_obsrv_vars.drop('gasmop', axis=1, inplace=True)
dp.rearrange_cols()
end_time = time.time()

print("Processing time (h:m:s) ", str(datetime.timedelta(seconds = round(end_time - start_time, 0))))

'''
Saving the generated training dataset into 'DATA' folder
'''
dp.ds_obsrv_vars.to_csv(path_or_buf="DATA/processed_dataset.csv", index=False)




