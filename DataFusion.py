import pandas as pd
from pandas import DataFrame
import os
import math
import numpy as np

class DataFusion():

    ds_obsrv_vars: DataFrame = None


    def initialise_dset_obsrv_vars(self, first100rows_only: bool = False) -> None:
        self.ds_obsrv_vars = pd.DataFrame()

        fuel_poverty_ds = pd.read_excel(io=os.path.join(os.path.dirname(__file__), r"DATA\RAW\Fuel_poverty_2015.xlsx"),
                                        sheet_name="fuel_poverty_2015_ukda")
        
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
            self.ds_obsrv_vars = self.ds_obsrv_vars[:500] # only keep first 100 rows
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

        self.ds_obsrv_vars['Y_0'] = self.ds_obsrv_vars.apply(lambda row: self.gas_cnsmp_IVW(row['V_6'], row['V_0'], row['V_3'], row['V_2'], row['V_7']), axis=1)
        return
    

    def gas_cnsmp_IVW(self, V_6_val, V_0_val, V_3_val, V_2_val, V_7_val) -> float:
        '''
        Given a series of observed values for the following variables:
         - V_6: dwelling floor area
         - V_0: dwelling type
         - V_3: dwelling age
         - V_2: tenancy
         - V_7: household income
        the method returns an Inverse-Variance Weighted mean estimate of annual energy (gas) consumption.
        '''
        V_7_val = self.V7_to_num(real_valued=V_7_val)

        fpath = os.path.join(os.path.dirname(__file__), r"DATA\RAW\Gas_consumption_data_2011.xlsx")

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

        weights = np.array([weight_V_6, weight_V_0, weight_V_3, weight_V_2, weight_V_7])
        means = np.array([gc_mean_given_V_6, gc_mean_given_V_0, gc_mean_given_V_3, gc_mean_given_V_2, gc_mean_given_V_7])

        gc_weighted_mean_val = np.sum(weights * means) / np.sum(weights)
        gc_weighted_variance_val = 1. / np.sum(weights)
        gc_weighted_std_dev_val = math.sqrt(gc_weighted_variance_val)

        sampled = np.random.normal(gc_weighted_mean_val, gc_weighted_std_dev_val) #instead of returning the mean we sample a random value from the combined distrib.

        return str(sampled, 4)
        #return str(round(gc_weighted_mean_val, 4))
    

    def fill_in_ind_temp_data(self) -> None:
        '''
        Fills in values of indoor temperature (Y_1) into the dataframe "ds_obsrv_vars"
        '''
        self.ds_obsrv_vars['Y_1'] = ""
        self.ds_obsrv_vars['Y_1'] = self.ds_obsrv_vars.apply(lambda row: self.indoord_tmpt_IVW(row['V_0'], row['V_3'], row['V_6'], row['X'], row['V_2'], row['V_5'], row['V_4']), axis=1)
        return
    

    def indoord_tmpt_IVW(self, V_0_val, V_3_val, V_6_val, X_val, V_2_val, V_5_val, V_4_val) -> float:
        '''
        Given a series of observed values for the following variables:
         - V_0: dwelling type
         - V_3: dwelling age
         - V_6: dwelling floor area
         - X:   walls insulation
         - V_2: tenancy
         - V_5: household size
         - V_4: under-occupancy
        the method returns an Inverse-Variance Weighted mean estimate of annual energy (gas) consumption.
        '''
        fpath = os.path.join(os.path.dirname(__file__), r"DATA\RAW\Energy_follow_Up_Survey_2011_mean_temp.xlsx")

        it_by_V_0 = pd.read_excel(io=fpath, sheet_name="by_dwelling_type")
        it_by_V_3 = pd.read_excel(io=fpath, sheet_name="by_dwelling_age")
        it_by_V_6 = pd.read_excel(io=fpath, sheet_name="by_floor_area")
        it_by_X = pd.read_excel(io=fpath, sheet_name="by_walls_insulation")
        it_by_V_2 = pd.read_excel(io=fpath, sheet_name="by_tenancy")
        it_by_V_5 = pd.read_excel(io=fpath, sheet_name="by_household_size")
        it_by_V_4 = pd.read_excel(io=fpath, sheet_name="by_under_occupancy")

        it_mean_given_V_0 = it_by_V_0.loc[it_by_V_0['DwellingType_value_num'] == V_0_val, 'Dwelling_mean_temp'].iloc[0]
        it_mean_given_V_3 = it_by_V_3.loc[it_by_V_3['DwellingAge_value_num'] == V_3_val, 'Dwelling_mean_temp'].iloc[0]
        it_mean_given_V_6 = it_by_V_6.loc[it_by_V_6['Floor_area_value_num'] == V_6_val, 'Dwelling_mean_temp'].iloc[0]
        it_mean_given_X   = it_by_X.loc[it_by_X['Walls_insulation_value_num'] == X_val, 'Dwelling_mean_temp'].iloc[0]
        it_mean_given_V_2 = it_by_V_2.loc[it_by_V_2['Tenancy_value_num'] == V_2_val, 'Dwelling_mean_temp'].iloc[0]
        it_mean_given_V_5 = it_by_V_5.loc[it_by_V_5['Household_size_value_num'] == V_5_val, 'Dwelling_mean_temp'].iloc[0]
        it_mean_given_V_4 = it_by_V_4.loc[it_by_V_4['Under_occupancy_value_num'] == V_4_val, 'Dwelling_mean_temp'].iloc[0]

        it_stdev_given_V_0 = it_by_V_0.loc[it_by_V_0['DwellingType_value_num'] == V_0_val, 'Dwelling_st_dev_temp'].iloc[0]
        it_stdev_given_V_3 = it_by_V_3.loc[it_by_V_3['DwellingAge_value_num'] == V_3_val, 'Dwelling_st_dev_temp'].iloc[0]
        it_stdev_given_V_6 = it_by_V_6.loc[it_by_V_6['Floor_area_value_num'] == V_6_val, 'Dwelling_st_dev_temp'].iloc[0]
        it_stdev_given_X   = it_by_X.loc[it_by_X['Walls_insulation_value_num'] == X_val, 'Dwelling_st_dev_temp'].iloc[0]
        it_stdev_given_V_2 = it_by_V_2.loc[it_by_V_2['Tenancy_value_num'] == V_2_val, 'Dwelling_st_dev_temp'].iloc[0]
        it_stdev_given_V_5 = it_by_V_5.loc[it_by_V_5['Household_size_value_num'] == V_5_val, 'Dwelling_st_dev_temp'].iloc[0]
        it_stdev_given_V_4 = it_by_V_4.loc[it_by_V_4['Under_occupancy_value_num'] == V_4_val, 'Dwelling_st_dev_temp'].iloc[0]

        weight_V_0 = 1. / math.pow(it_stdev_given_V_0, 2) # weight as inverse of Variance
        weight_V_3 = 1. / math.pow(it_stdev_given_V_3, 2)
        weight_V_6 = 1. / math.pow(it_stdev_given_V_6, 2)
        weight_X = 1. / math.pow(it_stdev_given_X, 2)
        weight_V_2 = 1. / math.pow(it_stdev_given_V_2, 2)
        weight_V_5 = 1. / math.pow(it_stdev_given_V_5, 2)
        weight_V_4 = 1. / math.pow(it_stdev_given_V_4, 2)

        weights = np.array([weight_V_0, weight_V_3, weight_V_6, weight_X, weight_V_2, weight_V_5, weight_V_4])
        means = np.array([it_mean_given_V_0, it_mean_given_V_3, it_mean_given_V_6, it_mean_given_X, it_mean_given_V_2, it_mean_given_V_5, it_mean_given_V_4])

        it_weighted_mean_val = np.sum(weights * means) / np.sum(weights)
        it_weighted_variance_val = 1. / np.sum(weights)
        it_weighted_std_dev_val = math.sqrt(it_weighted_variance_val)

        sampled = np.random.normal(it_weighted_mean_val, it_weighted_std_dev_val) #instead of returning the mean we sample a random value from the combined distrib.

        return str(sampled, 4) 
        #return str(round(it_weighted_mean_val, 4))
    



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


    




