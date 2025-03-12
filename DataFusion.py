import pandas as pd
from pandas import DataFrame, Series
import os
import math

class DataProcessing():

    ds_obsrv_vars: DataFrame = None


    def initialise_dset_obsrv_vars(self, first100rows_only: bool = False) -> None:
        self.ds_obsrv_vars = pd.DataFrame()

        fuel_poverty_ds = pd.read_excel(io=os.path.join(os.path.dirname(__file__), r"DATA\RAW\Fuel_poverty_2021.xlsx"),
                                        sheet_name="fuel_poverty_2021_ukda")
        
        self.ds_obsrv_vars.loc[:, 'X'] = fuel_poverty_ds.loc[:, 'WallType']
        self.ds_obsrv_vars.loc[:, 'V_0'] = fuel_poverty_ds.loc[:, 'DWtype']
        self.ds_obsrv_vars.loc[:, 'V_2'] = fuel_poverty_ds.loc[:, 'tenure4x']
        self.ds_obsrv_vars.loc[:, 'V_3'] = fuel_poverty_ds.loc[:, 'DWage']
        self.ds_obsrv_vars.loc[:, 'V_4'] = fuel_poverty_ds.loc[:, 'Unoc']
        self.ds_obsrv_vars.loc[:, 'V_5'] = fuel_poverty_ds.loc[:, 'Hhsize']
        self.ds_obsrv_vars.loc[:, 'V_6'] = fuel_poverty_ds.loc[:, 'FloorArea']
        self.ds_obsrv_vars.loc[:, 'V_7'] = fuel_poverty_ds.loc[:, 'fpfullinc']
        self.ds_obsrv_vars.loc[:, 'V_8'] = fuel_poverty_ds.loc[:, 'hhcompx']

        self.ds_obsrv_vars.loc[:, 'Mainfueltype'] = fuel_poverty_ds.loc[:, 'Mainfueltype']

        if first100rows_only is True:
            self.ds_obsrv_vars = self.ds_obsrv_vars[:100] # only keep first 100 rows
        return 
    

    def filter_for_main_fuel_type(self) -> None:
        '''
        Removes instances where main fuel type for space heating is not "Gas"
        '''
        self.ds_obsrv_vars = self.ds_obsrv_vars[self.ds_obsrv_vars.Mainfueltype != 2]
        self.ds_obsrv_vars = self.ds_obsrv_vars[self.ds_obsrv_vars.Mainfueltype != 3]
        return
    

    def filter_for_income(self) -> None:
        '''
        Removes instances (rows) where household income is bigger than £100k (as they are lumped all together 
        above that value in the original dataset) or smaller than £1k.
        '''
        self.ds_obsrv_vars = self.ds_obsrv_vars[self.ds_obsrv_vars.V_7 > 1000.]
        self.ds_obsrv_vars = self.ds_obsrv_vars[self.ds_obsrv_vars.V_7 < 99999.]
        return
    

    def aggregate_wall_type(self) -> None:
        '''Removes instances (rows) where wall type is "other".'''
        self.ds_obsrv_vars = self.ds_obsrv_vars[self.ds_obsrv_vars.X != 5]

        '''Aggregates cavity-insulated and solid-insulatedwalls (and cavity-uninsulated and solid-uninsulated)
        So that: 
            insulated walls = 1
            uninsulated walls = 2
        '''
        self.ds_obsrv_vars = self.ds_obsrv_vars.replace({'X': 3}, 1)
        self.ds_obsrv_vars = self.ds_obsrv_vars.replace({'X': 4}, 2)
        return
    

    def fill_in_gas_cnsmp_data(self) -> None:
        '''
        Fills in values of Gas energy use (Y_0) into the dataframe "ds_obsrv_vars"
        '''
        self.ds_obsrv_vars['Y_0'] = ""
        self.ds_obsrv_vars.drop('Mainfueltype', axis=1, inplace=True)

        self.ds_obsrv_vars['Y_0'] = self.ds_obsrv_vars.apply(lambda row: self.gas_cnsmp_IVW(row['V_6'], row['V_0'], row['V_3'], row['V_2'], row['V_7']), axis=1)
        return
    

    def gas_cnsmp_IVW(self, V_6_val, V_0_val, V_3_val, V_2_val, V_7_val) -> float:
        '''
        Given a series of observed values for variables V_6, V_0, V_3, V_2 and V_7
        the method returns an Inverse-Variance Weighted mean estimate of gas consumption.
        '''
        V_7_val = self.V7_to_num(real_valued=V_7_val)

        fpath = os.path.join(os.path.dirname(__file__), r"DATA\RAW\Gas_consumption_data_2021.xlsx")

        gc_by_V_6 = pd.read_excel(io=fpath, sheet_name="by_floor_area")
        gc_by_V_0 = pd.read_excel(io=fpath, sheet_name="by_dwelling_type")
        gc_by_V_3 = pd.read_excel(io=fpath, sheet_name="by_dwelling_age")
        gc_by_V_2 = pd.read_excel(io=fpath, sheet_name="by_tenancy")
        gc_by_V_7 = pd.read_excel(io=fpath, sheet_name="by_income")

        gc_mean_given_V_6 = gc_by_V_6.loc[gc_by_V_6['Floor_area_value_num'] == V_6_val, 'Gas_consumption_mean'].iloc[0]
        gc_mean_given_V_0 = gc_by_V_0.loc[gc_by_V_0['DwellingType_value_num'] == V_0_val, 'Gas_consumption_mean'].iloc[0]
        gc_mean_given_V_3 = gc_by_V_3.loc[gc_by_V_3['DwellingAge_value_num'] == V_3_val, 'Gas_consumption_mean'].iloc[0]
        gc_mean_given_V_2 = gc_by_V_2.loc[gc_by_V_2['Tenancy_value_num'] == V_2_val, 'Gas_consumption_mean'].iloc[0]
        gc_mean_given_V_7 = gc_by_V_7.loc[gc_by_V_7['Income_value_num'] == V_7_val, 'Gas_consumption_mean'].iloc[0]

        gc_stdev_given_V_6 = gc_by_V_6.loc[gc_by_V_6['Floor_area_value_num'] == V_6_val, 'Gas_consumption_st_dev'].iloc[0]
        gc_stdev_given_V_0 = gc_by_V_0.loc[gc_by_V_0['DwellingType_value_num'] == V_0_val, 'Gas_consumption_st_dev'].iloc[0]
        gc_stdev_given_V_3 = gc_by_V_3.loc[gc_by_V_3['DwellingAge_value_num'] == V_3_val, 'Gas_consumption_st_dev'].iloc[0]
        gc_stdev_given_V_2 = gc_by_V_2.loc[gc_by_V_2['Tenancy_value_num'] == V_2_val, 'Gas_consumption_st_dev'].iloc[0]
        gc_stdev_given_V_7 = gc_by_V_7.loc[gc_by_V_7['Income_value_num'] == V_7_val, 'Gas_consumption_st_dev'].iloc[0]
        
        weight_V_6 = 1. / math.pow(gc_stdev_given_V_6, 2) # weight as inverse of Variance
        weight_V_0 = 1. / math.pow(gc_stdev_given_V_0, 2)
        weight_V_3 = 1. / math.pow(gc_stdev_given_V_3, 2)
        weight_V_2 = 1. / math.pow(gc_stdev_given_V_2, 2)
        weight_V_7 = 1. / math.pow(gc_stdev_given_V_7, 2)

        gc_weighted_mean_val = ((weight_V_6 * gc_mean_given_V_6) + (weight_V_0 * gc_mean_given_V_0) + (weight_V_3 * gc_mean_given_V_3) + (weight_V_2 * gc_mean_given_V_2) + (weight_V_7 * gc_mean_given_V_7)) / (weight_V_6 + weight_V_0 + weight_V_3 + weight_V_2 + weight_V_7)
     

        return str(round(gc_weighted_mean_val, 1))
    

    def V7_to_num(self, real_valued: float) -> int:
        real_valued = float(real_valued)
        if real_valued < 15000:
            return 1
        elif real_valued <= 19999:
            return 2
        elif real_valued <= 29999:
            return 3
        elif real_valued <=39999:
            return 4
        elif real_valued <= 49999:
            return 5
        elif real_valued <= 59999:
            return 6
        elif real_valued <= 69000:
            return 7
        else:
            return 8


    




