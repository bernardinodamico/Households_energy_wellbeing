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

    gp_by_gasmop: DataFrame = None


    def __init__(self, subset_only: bool = False, how_many: int = 100):
        self.initialise_dset_obsrv_vars(subset_only=subset_only, how_many=how_many)
        self.initialise_gas_price_by_mop_dsets()
        
        return


    def initialise_gas_price_by_mop_dsets(self) -> None:
        fpath = os.path.join(os.path.dirname(__file__), r"DATA\RAW\Gas_price_per_kWh_2015.xlsx")
        self.gp_by_gasmop = pd.read_excel(io=fpath, sheet_name="2015_gas_price_per_kWh")
        return


    def initialise_dset_obsrv_vars(self, subset_only: bool = False, how_many: int = 100) -> None:
        
        self.ds_obsrv_vars = pd.DataFrame()

        fuel_poverty_ds = pd.read_excel(io=os.path.join(os.path.dirname(__file__), r"DATA\RAW\English_Housing_Survey_FuelP_dataset_2015.xlsx"),
                                        sheet_name="fuel_poverty_2015_ukda")
        
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

        if subset_only is True:
            self.ds_obsrv_vars = self.ds_obsrv_vars[:how_many] # only keep first 100 rows
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
        self.ds_obsrv_vars = self.ds_obsrv_vars[self.ds_obsrv_vars.W < 0.3]
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


    def discretise(self) -> None:

        self.discrete_ds_obsrv_vars = self.ds_obsrv_vars.copy(deep=True)

        self.discrete_ds_obsrv_vars['Y_0'] = pd.cut(self.discrete_ds_obsrv_vars['Y_0'],
               bins=GetVariableValues.get_bins_intervals(var_symbol='Y_0'),
               labels=GetVariableValues.get_nums(var_symbol='Y_0')
               )
        
        self.discrete_ds_obsrv_vars['W'] = pd.cut(self.discrete_ds_obsrv_vars['W'],
               bins=GetVariableValues.get_bins_intervals(var_symbol='W'),
               labels=GetVariableValues.get_nums(var_symbol='W')
               )
        
        self.discrete_ds_obsrv_vars['V_1'] = pd.cut(self.discrete_ds_obsrv_vars['V_1'],
               bins=GetVariableValues.get_bins_intervals(var_symbol='V_1'),
               labels=GetVariableValues.get_nums(var_symbol='V_1')
               )
        
        self.discrete_ds_obsrv_vars['V_7'] = pd.cut(self.discrete_ds_obsrv_vars['V_7'],
               bins=GetVariableValues.get_bins_intervals(var_symbol='V_7'),
               labels=GetVariableValues.get_nums(var_symbol='V_7')
               )

        return



'''
Below is a function instantiating the DataFusion class. It builds the dataset 
for training the model parameters of the Causal Bayesian Network.
'''

def gen_training_dataset():

    start_time = time.time()
    dp = DataFusion(subset_only=False, how_many=None)

    dp.filter_for_main_fuel_type()
    dp.filter_for_method_of_payment()
    dp.filter_for_income()

    dp.aggregate_wall_type()

    dp.fill_in_gas_cnsmp_data() 
    dp.fill_in_energy_burden_data()
    dp.fill_in_W_binary(fuel_poverty_treshold=0.1)
    dp.filter_for_en_burden()

    dp.ds_obsrv_vars.drop('Mainfueltype', axis=1, inplace=True)
    dp.ds_obsrv_vars.drop('gasmop', axis=1, inplace=True)
    dp.ds_obsrv_vars.drop('litecost', axis=1, inplace=True)

    dp.rearrange_cols()
    dp.discretise() 

    # save processed datasets to csv files
    dp.ds_obsrv_vars.to_csv(path_or_buf="DATA/processed_dataset.csv", index=False)
    dp.discrete_ds_obsrv_vars.to_csv(path_or_buf="DATA/discretised_processed_dataset.csv", index=False)
    end_time = time.time()

    print("Processing time (h:m:s) ", str(datetime.timedelta(seconds = round(end_time - start_time, 0))))

    return