import pandas as pd
from pandas import DataFrame
import os
import math
import numpy as np
from Values_mapping import GetVariableValues

class DataFusion():

    ds_obsrv_vars: DataFrame = None
    discrete_ds_obsrv_vars: DataFrame = None


    it_by_V_0: DataFrame = None
    it_by_V_3: DataFrame = None
    it_by_V_6: DataFrame = None
    it_by_X: DataFrame = None
    it_by_V_2: DataFrame = None
    it_by_V_5: DataFrame = None
    it_by_V_4: DataFrame = None

    gc_by_V_6: DataFrame = None
    gc_by_V_0: DataFrame = None
    gc_by_V_3: DataFrame = None
    gc_by_V_2: DataFrame = None
    gc_by_V_7: DataFrame = None
    gc_by_X: DataFrame = None
    gp_by_gasmop: DataFrame = None

    def __init__(self, subset_only: bool = False, how_many: int = 100):
        self.initialise_dset_obsrv_vars(subset_only=subset_only, how_many=how_many)
        self.initialise_indoor_temp_dsets()
        self.initialise_gas_cnsmp_dsets()
        self.initialise_gas_price_by_mop_dsets()
        
        return


    def initialise_gas_price_by_mop_dsets(self) -> None:
        fpath = os.path.join(os.path.dirname(__file__), r"DATA\RAW\Gas_price_per_kWh_2015.xlsx")
        self.gp_by_gasmop = pd.read_excel(io=fpath, sheet_name="2015_gas_price_per_kWh")
        return


    def initialise_gas_cnsmp_dsets(self) -> None:
        fpath = os.path.join(os.path.dirname(__file__), r"DATA\RAW\Gas_consumption_data_2015.xlsx")

        self.gc_by_V_6 = pd.read_excel(io=fpath, sheet_name="by_floor_area")
        self.gc_by_V_0 = pd.read_excel(io=fpath, sheet_name="by_dwelling_type")
        self.gc_by_V_3 = pd.read_excel(io=fpath, sheet_name="by_dwelling_age")
        self.gc_by_V_2 = pd.read_excel(io=fpath, sheet_name="by_tenancy")
        self.gc_by_V_7 = pd.read_excel(io=fpath, sheet_name="by_income")
        self.gc_by_X = pd.read_excel(io=fpath, sheet_name="by_walls_insulation")

        return


    def initialise_indoor_temp_dsets(self) -> None:
        fpath = os.path.join(os.path.dirname(__file__), r"DATA\RAW\Energy_follow_Up_Survey_2011_mean_temp.xlsx")
        
        self.it_by_V_0 = pd.read_excel(io=fpath, sheet_name="by_dwelling_type")
        self.it_by_V_3 = pd.read_excel(io=fpath, sheet_name="by_dwelling_age")
        self.it_by_V_6 = pd.read_excel(io=fpath, sheet_name="by_floor_area")
        self.it_by_X = pd.read_excel(io=fpath, sheet_name="by_walls_insulation")
        self.it_by_V_2 = pd.read_excel(io=fpath, sheet_name="by_tenancy")
        self.it_by_V_5 = pd.read_excel(io=fpath, sheet_name="by_household_size")
        self.it_by_V_4 = pd.read_excel(io=fpath, sheet_name="by_under_occupancy")

        return


    def initialise_dset_obsrv_vars(self, subset_only: bool = False, how_many: int = 100) -> None:
        
        self.ds_obsrv_vars = pd.DataFrame()

        fuel_poverty_ds = pd.read_excel(io=os.path.join(os.path.dirname(__file__), r"DATA\RAW\English_Housing_Survey_FuelP_dataset_2015.xlsx"),
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
        Removes instances (rows) where household income is smaller than £1000.
        '''
        self.ds_obsrv_vars = self.ds_obsrv_vars[self.ds_obsrv_vars.V_7 > 1000.]
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

        self.ds_obsrv_vars['Y_0'] = self.ds_obsrv_vars.apply(lambda row: self._gas_cnsmp_IVW(row['V_6'], row['V_0'], row['V_3'], row['V_2'], row['V_7'], row['X']), axis=1)
        return
    

    def _gas_cnsmp_IVW(self, V_6_val, V_0_val, V_3_val, V_2_val, V_7_val, X_val) -> float:
        '''
        Given a series of observed values for the following variables:
         - V_6: dwelling floor area
         - V_0: dwelling type
         - V_3: dwelling age
         - V_2: tenancy
         - V_7: household income
         - X: walls insulaton
        the method returns an Inverse-Variance Weighted mean estimate of annual energy (gas) consumption.
        '''
        V_7_val = self._V7_to_num(real_valued=V_7_val)

        gc_mean_given_V_6 = self.gc_by_V_6.loc[self.gc_by_V_6['Floor_area_value_num'] == V_6_val, 'Gas_consumption_mean'].iloc[0]
        gc_mean_given_V_0 = self.gc_by_V_0.loc[self.gc_by_V_0['DwellingType_value_num'] == V_0_val, 'Gas_consumption_mean'].iloc[0]
        gc_mean_given_V_3 = self.gc_by_V_3.loc[self.gc_by_V_3['DwellingAge_value_num'] == V_3_val, 'Gas_consumption_mean'].iloc[0]
        gc_mean_given_V_2 = self.gc_by_V_2.loc[self.gc_by_V_2['Tenancy_value_num'] == V_2_val, 'Gas_consumption_mean'].iloc[0]
        gc_mean_given_V_7 = self.gc_by_V_7.loc[self.gc_by_V_7['Income_value_num'] == V_7_val, 'Gas_consumption_mean'].iloc[0]
        gc_mean_given_X = self.gc_by_X.loc[self.gc_by_X['Walls_insulation_value_num'] == X_val, 'Gas_consumption_mean'].iloc[0]

        gc_stdev_given_V_6 = self.gc_by_V_6.loc[self.gc_by_V_6['Floor_area_value_num'] == V_6_val, 'Gas_consumption_st_dev'].iloc[0]
        gc_stdev_given_V_0 = self.gc_by_V_0.loc[self.gc_by_V_0['DwellingType_value_num'] == V_0_val, 'Gas_consumption_st_dev'].iloc[0]
        gc_stdev_given_V_3 = self.gc_by_V_3.loc[self.gc_by_V_3['DwellingAge_value_num'] == V_3_val, 'Gas_consumption_st_dev'].iloc[0]
        gc_stdev_given_V_2 = self.gc_by_V_2.loc[self.gc_by_V_2['Tenancy_value_num'] == V_2_val, 'Gas_consumption_st_dev'].iloc[0]
        gc_stdev_given_V_7 = self.gc_by_V_7.loc[self.gc_by_V_7['Income_value_num'] == V_7_val, 'Gas_consumption_st_dev'].iloc[0]
        gc_stdev_given_X = self.gc_by_X.loc[self.gc_by_X['Walls_insulation_value_num'] == X_val, 'Gas_consumption_st_dev'].iloc[0]
        
        weight_V_6 = 1. / math.pow(gc_stdev_given_V_6, 2) # weight as inverse of Variance
        weight_V_0 = 1. / math.pow(gc_stdev_given_V_0, 2)
        weight_V_3 = 1. / math.pow(gc_stdev_given_V_3, 2)
        weight_V_2 = 1. / math.pow(gc_stdev_given_V_2, 2)
        weight_V_7 = 1. / math.pow(gc_stdev_given_V_7, 2)
        weight_X = 1. / math.pow(gc_stdev_given_V_7, 2)

        weights = np.array([weight_X, weight_V_7])#, weight_V_6, weight_V_2, weight_V_0, weight_V_3])
        means = np.array([gc_mean_given_X, gc_mean_given_V_7])# gc_mean_given_V_6#, gc_mean_given_V_2, gc_mean_given_V_0, gc_mean_given_V_3])
        st_devs = np.array([gc_stdev_given_V_6, gc_stdev_given_V_0, gc_stdev_given_V_3, gc_stdev_given_V_2, gc_stdev_given_V_7, gc_stdev_given_X])


        gc_weighted_mean_val = np.sum(weights * means) / np.sum(weights)
        #gc_weighted_std_dev_val = np.sum(weights * st_devs) / np.sum(weights)

        #sampled = np.random.normal(gc_weighted_mean_val, gc_weighted_std_dev_val) #instead of returning the mean we sample a random value from the combined distrib.

        #return round(sampled, 1)
        return round(gc_weighted_mean_val, 1)
    

    def fill_in_ind_temp_data(self) -> None:
        '''
        Fills in values of indoor temperature (Y_1) into the dataframe "ds_obsrv_vars"
        '''
        self.ds_obsrv_vars['Y_1'] = ""
        self.ds_obsrv_vars['Y_1'] = self.ds_obsrv_vars.apply(lambda row: self._indoord_tmpt_IVW(row['V_0'], row['V_3'], row['V_6'], row['X'], row['V_2'], row['V_5'], row['V_4']), axis=1)
        return
    

    def _indoord_tmpt_IVW(self, V_0_val, V_3_val, V_6_val, X_val, V_2_val, V_5_val, V_4_val) -> float:
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

        it_mean_given_V_0 = self.it_by_V_0.loc[self.it_by_V_0['DwellingType_value_num'] == V_0_val, 'Dwelling_mean_temp'].iloc[0]
        it_mean_given_V_3 = self.it_by_V_3.loc[self.it_by_V_3['DwellingAge_value_num'] == V_3_val, 'Dwelling_mean_temp'].iloc[0]
        it_mean_given_V_6 = self.it_by_V_6.loc[self.it_by_V_6['Floor_area_value_num'] == V_6_val, 'Dwelling_mean_temp'].iloc[0]
        it_mean_given_X   = self.it_by_X.loc[self.it_by_X['Walls_insulation_value_num'] == X_val, 'Dwelling_mean_temp'].iloc[0]
        it_mean_given_V_2 = self.it_by_V_2.loc[self.it_by_V_2['Tenancy_value_num'] == V_2_val, 'Dwelling_mean_temp'].iloc[0]
        it_mean_given_V_5 = self.it_by_V_5.loc[self.it_by_V_5['Household_size_value_num'] == V_5_val, 'Dwelling_mean_temp'].iloc[0]
        it_mean_given_V_4 = self.it_by_V_4.loc[self.it_by_V_4['Under_occupancy_value_num'] == V_4_val, 'Dwelling_mean_temp'].iloc[0]

        it_stdev_given_V_0 = self.it_by_V_0.loc[self.it_by_V_0['DwellingType_value_num'] == V_0_val, 'Dwelling_st_dev_temp'].iloc[0]
        it_stdev_given_V_3 = self.it_by_V_3.loc[self.it_by_V_3['DwellingAge_value_num'] == V_3_val, 'Dwelling_st_dev_temp'].iloc[0]
        it_stdev_given_V_6 = self.it_by_V_6.loc[self.it_by_V_6['Floor_area_value_num'] == V_6_val, 'Dwelling_st_dev_temp'].iloc[0]
        it_stdev_given_X   = self.it_by_X.loc[self.it_by_X['Walls_insulation_value_num'] == X_val, 'Dwelling_st_dev_temp'].iloc[0]
        it_stdev_given_V_2 = self.it_by_V_2.loc[self.it_by_V_2['Tenancy_value_num'] == V_2_val, 'Dwelling_st_dev_temp'].iloc[0]
        it_stdev_given_V_5 = self.it_by_V_5.loc[self.it_by_V_5['Household_size_value_num'] == V_5_val, 'Dwelling_st_dev_temp'].iloc[0]
        it_stdev_given_V_4 = self.it_by_V_4.loc[self.it_by_V_4['Under_occupancy_value_num'] == V_4_val, 'Dwelling_st_dev_temp'].iloc[0]

        weight_V_0 = 1. / math.pow(it_stdev_given_V_0, 2) # weight as inverse of Variance
        weight_V_3 = 1. / math.pow(it_stdev_given_V_3, 2)
        weight_V_6 = 1. / math.pow(it_stdev_given_V_6, 2)
        weight_X = 1. / math.pow(it_stdev_given_X, 2)
        weight_V_2 = 1. / math.pow(it_stdev_given_V_2, 2)
        weight_V_5 = 1. / math.pow(it_stdev_given_V_5, 2)
        weight_V_4 = 1. / math.pow(it_stdev_given_V_4, 2)

        weights = np.array([weight_V_0, weight_V_3, weight_V_6, weight_X, weight_V_2, weight_V_5, weight_V_4])
        means = np.array([it_mean_given_V_0, it_mean_given_V_3, it_mean_given_V_6, it_mean_given_X, it_mean_given_V_2, it_mean_given_V_5, it_mean_given_V_4])
        st_devs = np.array([it_stdev_given_V_0, it_stdev_given_V_3, it_stdev_given_V_6, it_stdev_given_X, it_stdev_given_V_2, it_stdev_given_V_5, it_stdev_given_V_4])

        it_weighted_mean_val = np.sum(weights * means) / np.sum(weights)
        it_weighted_std_dev_val = np.sum(weights * st_devs) / np.sum(weights)

        sampled = np.random.normal(it_weighted_mean_val, it_weighted_std_dev_val) #instead of returning the mean we sample a random value from the combined distrib.

        #return round(sampled, 4) 
        return round(it_weighted_mean_val, 4)
    

    def _V7_to_num(self, real_valued: float) -> int:
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
        elif real_valued <= 69999:
            return 7
        elif real_valued <= 99999:
            return 8
        else:
            return 9


    def fill_in_gas_cost_data(self) -> None:
            '''
            Fills in values of annual energy (gas) cost [£/year] (V_1) into the dataframe "ds_obsrv_vars"
            '''
            self.ds_obsrv_vars['V_1'] = ""
            self.ds_obsrv_vars['V_1'] = self.ds_obsrv_vars.apply(lambda row: round(self._gas_cost(row['gasmop']) * row['Y_0'], 1), axis=1)
       
            return
    

    def _gas_cost(self, gasmop_val) -> float:
        '''
        Given the observed value for the following variable: 
        - gasmop: method of payment for gas
        the function returns a value for the energy (gas) price [£/kWh] variable.
        '''

        gp_given_gasmop = self.gp_by_gasmop.loc[self.gp_by_gasmop['Payment_method_value_num'] == gasmop_val, 'Annual gas price per 1 kWh'].iloc[0]

        return gp_given_gasmop
    

    def fill_in_energy_burden_data(self) -> None:
        '''
        Fills in values of energy burden [£/£] (W) into the dataframe "ds_obsrv_vars"
        NOTE: the total energy cost (gas + electricity) is used to calculate  energy burden.
        '''
        self.ds_obsrv_vars['W'] = ""
        self.ds_obsrv_vars['W'] = round((self.ds_obsrv_vars['V_1'] + self.ds_obsrv_vars['litecost']) / self.ds_obsrv_vars['V_7'], 4)
    
        return
    

    def rearrange_cols(self) -> None:
        self.ds_obsrv_vars = self.ds_obsrv_vars[['X', 'Y_0', 'Y_1', 'W', 'F_p', 'V_0', 'V_1', 'V_2', 'V_3', 'V_4', 'V_5', 'V_6', 'V_7', 'V_8']]
        return
    

    def filter_for_en_burden(self) -> None:
        '''
        Removes instances (rows) where energy_burden is biggher than a treshold.
        '''
        self.ds_obsrv_vars = self.ds_obsrv_vars[self.ds_obsrv_vars.W < 0.3]
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
               labels=['0', '1']
               )
        return


    def discretise(self) -> None:

        self.discrete_ds_obsrv_vars = self.ds_obsrv_vars.copy(deep=True)

        self.discrete_ds_obsrv_vars['Y_0'] = pd.cut(self.discrete_ds_obsrv_vars['Y_0'],
               bins=GetVariableValues.get_bins_intervals(var_symbol='Y_0'),
               labels=GetVariableValues.get_nums(var_symbol='Y_0')
               )
        
        self.discrete_ds_obsrv_vars['Y_1'] = pd.cut(self.discrete_ds_obsrv_vars['Y_1'],
               bins=GetVariableValues.get_bins_intervals(var_symbol='Y_1'),
               labels=GetVariableValues.get_nums(var_symbol='Y_1')
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

