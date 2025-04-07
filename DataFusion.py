import pandas as pd
from pandas import DataFrame
import os
import math
import numpy as np
from Values_mapping import GetVariableValues
import time
import datetime



class DataFusion():

    ds_obsrv_vars: DataFrame = None
    discrete_ds_obsrv_vars: DataFrame = None
    ref_year: int = None 
    gp_by_gasmop: DataFrame = None
    weight_factor_name: str = None


    def __init__(self, year: int, subset_only: bool = False, how_many: int = 100):
        '''
        'year' parameter: any of the following: 2015, 2016, 2017, 2018
        '''
        self.ref_year = year
        self._fetch_name_cols()
        self.initialise_dset_obsrv_vars(subset_only=subset_only, how_many=how_many)
        self.initialise_gas_price_by_mop_dsets()

        return


    def initialise_gas_price_by_mop_dsets(self) -> None:
        fpath = os.path.join(os.path.dirname(__file__), r"DATA\RAW\Gas_price_per_kWh.xlsx")
        self.gp_by_gasmop = pd.read_excel(io=fpath, sheet_name=f"{self.ref_year}_gas_price_per_kWh")
        return


    def initialise_dset_obsrv_vars(self, subset_only: bool = False, how_many: int = 100) -> None:
        
        self.ds_obsrv_vars = pd.DataFrame()

        fuel_poverty_ds = pd.read_excel(io=os.path.join(os.path.dirname(__file__), r"DATA\RAW\English_Housing_Survey_FuelP_dataset.xlsx"),
                                        sheet_name=f"fuel_poverty_{self.ref_year}_ukda")
        
        self.ds_obsrv_vars.loc[:, 'X'] = fuel_poverty_ds.loc[:, 'WallType']
        self.ds_obsrv_vars.loc[:, 'V_0'] = fuel_poverty_ds.loc[:, 'DWtype']
        self.ds_obsrv_vars.loc[:, 'V_1'] = fuel_poverty_ds.loc[:, 'spahcost']
        self.ds_obsrv_vars.loc[:, 'V_2'] = fuel_poverty_ds.loc[:, 'tenure4x']
        self.ds_obsrv_vars.loc[:, 'V_3'] = fuel_poverty_ds.loc[:, 'DWage']
        self.ds_obsrv_vars.loc[:, 'V_4'] = fuel_poverty_ds.loc[:, 'Unoc']
        self.ds_obsrv_vars.loc[:, 'V_5'] = fuel_poverty_ds.loc[:, 'Hhsize']
        self.ds_obsrv_vars.loc[:, 'V_6'] = fuel_poverty_ds.loc[:, 'FloorArea']
        self.ds_obsrv_vars.loc[:, 'V_7'] = fuel_poverty_ds.loc[:, 'fpfullinc']
        self.ds_obsrv_vars.loc[:, 'V_8'] = fuel_poverty_ds.loc[:, 'hhcompx']

        self.ds_obsrv_vars.loc[:, 'Mainfueltype'] = fuel_poverty_ds.loc[:, 'Mainfueltype'] # Main fule type variable
        self.ds_obsrv_vars.loc[:, 'gasmop'] = fuel_poverty_ds.loc[:, 'gasmop'] # Method of payment for gas {1: Direct debit; 2: Standard credit; 3: Pre payment} 
        self.ds_obsrv_vars.loc[:, 'litecost'] = fuel_poverty_ds.loc[:, 'litecost'] # Annual cost (£) to the household of powering their lights and appliances
        self.ds_obsrv_vars.loc[:, self.weight_factor_name] = fuel_poverty_ds.loc[:, self.weight_factor_name] # household weight factor indicating number of units like this one in the entire English domestic building stock and household population 

        if subset_only is True:
            self.ds_obsrv_vars = self.ds_obsrv_vars[:how_many] # only keep first n rows (n=how_many)
        return 
    
    def _fetch_name_cols(self) -> None:
        '''
        EHS Fuel Povery datasets use different column heading for some variables
        depedning on the Year release. The method fetches the correct string name.
        '''
        if self.ref_year == 2014:
            self.weight_factor_name = 'aagph1314'
        elif self.ref_year == 2015:
            self.weight_factor_name = 'aagph1415'
        elif self.ref_year == 2016:
            self.weight_factor_name = 'aagph1516'
        elif self.ref_year == 2017:
            self.weight_factor_name = 'aagph1617'
        elif self.ref_year == 2018:
            self.weight_factor_name = 'aagph1718'
        elif self.ref_year == 2019:
            self.weight_factor_name = 'aagph1819'
        elif self.ref_year == 2020:
            self.weight_factor_name = 'aagph1920'
        elif self.ref_year == 2021:
            self.weight_factor_name = 'aagph2021'

        return


    def filter_for_main_fuel_type(self) -> None:
        '''
        Removes instances where main fuel type for space heating is not "Gas"
        '''
        self.ds_obsrv_vars = self.ds_obsrv_vars[self.ds_obsrv_vars.Mainfueltype != 2]
        self.ds_obsrv_vars = self.ds_obsrv_vars[self.ds_obsrv_vars.Mainfueltype != 3]
        return
    

    def filter_for_method_of_payment(self) -> None:
        '''
        Removes instances where method of payment for gas is n/a. 
        '''
        self.ds_obsrv_vars = self.ds_obsrv_vars[self.ds_obsrv_vars.gasmop != 88]
        return
    

    def filter_for_income(self) -> None:
        '''
        Removes instances (rows) for household income outlyers.
        '''
        self.ds_obsrv_vars = self.ds_obsrv_vars[self.ds_obsrv_vars.V_7 > 5000.]
        self.ds_obsrv_vars = self.ds_obsrv_vars[self.ds_obsrv_vars.V_7 < 99999.]
        return
    

    def aggregate_wall_type(self) -> None:
        '''Removes instances (rows) where wall type is "other".'''
        self.ds_obsrv_vars = self.ds_obsrv_vars[self.ds_obsrv_vars.X != 5]

        '''Aggregates cavity-insulated and solid-insulatedwalls (and cavity-uninsulated and solid-uninsulated)
        So that: 
            Uninsulated walls = 1
            Insulated walls = 2
        '''
        self.ds_obsrv_vars = self.ds_obsrv_vars.replace({'X': 3}, 1)
        self.ds_obsrv_vars = self.ds_obsrv_vars.replace({'X': 4}, 2)
        return
    

    def fill_in_gas_cnsmp_data(self) -> None:
        '''
        Fills in values of Gas energy use (Y_0) into the dataframe "ds_obsrv_vars"
        '''
        self.ds_obsrv_vars['Y_0'] = ""

        self.ds_obsrv_vars['Y_0'] = self.ds_obsrv_vars.apply(lambda row: self._gas_cnsmp(row['V_1'], row['gasmop']), axis=1)
        return
    

    def _gas_cnsmp(self, V_1_val, gmop_val) -> float:
        '''
        The method returns a value for energy (gas) consumption based on the household heating (gas) cost (V_1) and gas price, which
        is dependend on the method of payment: direct-debit, standard credit, pre-payment 
        '''
        gprice_given_gmop = self.gp_by_gasmop.loc[self.gp_by_gasmop['Payment_method_value_num'] == gmop_val, 'Annual gas price per 1 kWh'].iloc[0]
        gas_consumtion = V_1_val / gprice_given_gmop

        return round(gas_consumtion, 1)
    

    def fill_in_energy_burden_data(self) -> None:
        '''
        Fills in values of energy burden [£/£] (W) into the dataframe "ds_obsrv_vars"
        NOTE: the total energy cost (gas + electricity) is used to calculate  energy burden.
        '''
        self.ds_obsrv_vars['W'] = ""
        self.ds_obsrv_vars['W'] = round((self.ds_obsrv_vars['V_1'] + self.ds_obsrv_vars['litecost']) / self.ds_obsrv_vars['V_7'], 4)
    
        return
    

    def rearrange_cols(self) -> None:
        self.ds_obsrv_vars = self.ds_obsrv_vars[['X', 'Y_0', 'W', 'F_p', 'V_0', 'V_1', 'V_2', 'V_3', 'V_4', 'V_5', 'V_6', 'V_7', 'V_8']]
        return
    

    def filter_for_en_burden(self) -> None:
        '''
        Removes instances (rows) where energy_burden is biggher (smaller) than a treshold.
        '''
        self.ds_obsrv_vars = self.ds_obsrv_vars[self.ds_obsrv_vars.W < 0.13]
        self.ds_obsrv_vars = self.ds_obsrv_vars[self.ds_obsrv_vars.W > 0.01]
        return


    def fill_in_W_binary(self, fuel_poverty_treshold: float = 0.1) -> None:
        '''
        Add a secondary variable for fuel poverty discretised as a binary:
            F_p = 0 --> "Not fuel poor"
            F_p = 1 --> "Fuel poor"
        '''
        self.ds_obsrv_vars['F_p'] = self.ds_obsrv_vars['W']
        self.ds_obsrv_vars['F_p'] = pd.cut(self.ds_obsrv_vars['F_p'],
               bins=[0, fuel_poverty_treshold, 1],
               labels=[0, 1]
               )
        return


    def discretise(self, Y_0_bins_num: int, W_bins_num: int, V_1_bins_num: int, V_7_bins_num: int) -> None:

        self.discrete_ds_obsrv_vars = self.ds_obsrv_vars.copy(deep=True)

        self.discrete_ds_obsrv_vars['Y_0'] = pd.cut(self.discrete_ds_obsrv_vars['Y_0'],
               bins=GetVariableValues.get_bins_intervals(var_symbol='Y_0', Y0bn=Y_0_bins_num, Wbn=W_bins_num, V1bn=V_1_bins_num, V7bn=V_7_bins_num),
               labels=GetVariableValues.get_nums(var_symbol='Y_0', Y0bn=Y_0_bins_num, Wbn=W_bins_num, V1bn=V_1_bins_num, V7bn=V_7_bins_num)
               )
        
        self.discrete_ds_obsrv_vars['W'] = pd.cut(self.discrete_ds_obsrv_vars['W'],
               bins=GetVariableValues.get_bins_intervals(var_symbol='W', Y0bn=Y_0_bins_num, Wbn=W_bins_num, V1bn=V_1_bins_num, V7bn=V_7_bins_num),
               labels=GetVariableValues.get_nums(var_symbol='W', Y0bn=Y_0_bins_num, Wbn=W_bins_num, V1bn=V_1_bins_num, V7bn=V_7_bins_num)
               )
        
        self.discrete_ds_obsrv_vars['V_1'] = pd.cut(self.discrete_ds_obsrv_vars['V_1'],
               bins=GetVariableValues.get_bins_intervals(var_symbol='V_1', Y0bn=Y_0_bins_num, Wbn=W_bins_num, V1bn=V_1_bins_num, V7bn=V_7_bins_num),
               labels=GetVariableValues.get_nums(var_symbol='V_1', Y0bn=Y_0_bins_num, Wbn=W_bins_num, V1bn=V_1_bins_num, V7bn=V_7_bins_num)
               )
        
        self.discrete_ds_obsrv_vars['V_7'] = pd.cut(self.discrete_ds_obsrv_vars['V_7'],
               bins=GetVariableValues.get_bins_intervals(var_symbol='V_7', Y0bn=Y_0_bins_num, Wbn=W_bins_num, V1bn=V_1_bins_num, V7bn=V_7_bins_num),
               labels=GetVariableValues.get_nums(var_symbol='V_7', Y0bn=Y_0_bins_num, Wbn=W_bins_num, V1bn=V_1_bins_num, V7bn=V_7_bins_num)
               )

        return
    

    def weighted_resampling(self, sample_size: int) -> None:
        '''
        The method uses the 'aagph1415' weight value to generate a re-sampled dataset by drawing from
        the existing dataset based on the weight values assignded to each unir (dataset row).
        By doing so, the returned sample dataset is representative of the English household and dwelling stock
        - sample_size parameter is the total number of draws
        '''
        sampled_df = self.ds_obsrv_vars.sample(n=sample_size, weights=self.weight_factor_name, random_state=42, axis=0, replace=True)

        self.ds_obsrv_vars = sampled_df 
        return



'''
Below is a function instantiating the DataFusion class. It builds the dataset 
for training the model parameters of the Causal Bayesian Network.
'''

def gen_training_dataset(Y_0_bins_num: int, W_bins_num: int, V_1_bins_num: int, V_7_bins_num: int):

    start_time = time.time()
    combined_ds_obsrv_vars = pd.DataFrame()
    combined_discrete_ds_obsrv_vars = pd.DataFrame()
    
    for ref_year in range(2015, 2019): # i.e. the 4-year time period from 2015 to 2018 
        dp = DataFusion(year=ref_year, subset_only=False, how_many=None)

        dp.filter_for_main_fuel_type()
        dp.filter_for_method_of_payment()
        dp.filter_for_income()

        dp.aggregate_wall_type()

        dp.fill_in_gas_cnsmp_data() 
        dp.fill_in_energy_burden_data()
        dp.fill_in_W_binary(fuel_poverty_treshold=0.1)
        dp.filter_for_en_burden()

        dp.weighted_resampling(sample_size=60000)

        dp.ds_obsrv_vars.drop('Mainfueltype', axis=1, inplace=True)
        dp.ds_obsrv_vars.drop('gasmop', axis=1, inplace=True)
        dp.ds_obsrv_vars.drop('litecost', axis=1, inplace=True)
        dp.ds_obsrv_vars.drop(dp.weight_factor_name, axis=1, inplace=True)

        dp.rearrange_cols()
        dp.discretise(Y_0_bins_num=Y_0_bins_num, W_bins_num=W_bins_num, V_1_bins_num=V_1_bins_num, V_7_bins_num=V_7_bins_num) 

        combined_ds_obsrv_vars = pd.concat([combined_ds_obsrv_vars, dp.ds_obsrv_vars], ignore_index=True)
        combined_discrete_ds_obsrv_vars = pd.concat([combined_discrete_ds_obsrv_vars, dp.discrete_ds_obsrv_vars], ignore_index=True)

    # save processed datasets to csv files   
    combined_ds_obsrv_vars.to_csv(path_or_buf="DATA/processed_dataset.csv", index=False)
    combined_discrete_ds_obsrv_vars.to_csv(path_or_buf="DATA/discretised_processed_dataset.csv", index=False)
    
    end_time = time.time()

    print("Data pre-processing time (h:m:s) ", str(datetime.timedelta(seconds = round(end_time - start_time, 0))))

    return combined_discrete_ds_obsrv_vars