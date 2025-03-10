import pandas as pd
from pandas import DataFrame
import os

class DataProcessing():

    ds_obsrv_vars: DataFrame = None


    
    def initialise_dset_obsrv_vars(self) -> None:
        self.ds_obsrv_vars = pd.DataFrame()

        fuel_poverty_ds = pd.read_excel(io=os.path.join(os.path.dirname(__file__), r"DATA\RAW\Fuel_poverty_2021.xlsx"),
                                        sheet_name="fuel_poverty_2021_ukda")
        
        self.ds_obsrv_vars.loc[:, 'X'] = fuel_poverty_ds.loc[:, 'WallType']
        self.ds_obsrv_vars.loc[:, 'V_0'] = fuel_poverty_ds.loc[:, 'DWtype']
        self.ds_obsrv_vars.loc[:, 'V_2'] = fuel_poverty_ds.loc[:, 'tenure4x']
        self.ds_obsrv_vars.loc[:, 'V_3'] = fuel_poverty_ds.loc[:, 'DWage']
        self.ds_obsrv_vars.loc[:, 'V_5'] = fuel_poverty_ds.loc[:, 'Hhsize']
        self.ds_obsrv_vars.loc[:, 'V_6'] = fuel_poverty_ds.loc[:, 'FloorArea']
        self.ds_obsrv_vars.loc[:, 'V_7'] = fuel_poverty_ds.loc[:, 'fpfullinc']
        self.ds_obsrv_vars.loc[:, 'V_8'] = fuel_poverty_ds.loc[:, 'hhcompx']

        self.ds_obsrv_vars.loc[:, 'Mainfueltype'] = fuel_poverty_ds.loc[:, 'Mainfueltype']
        return 
    

    def filter_for_main_fuel_type(self) -> None:
        '''
        Removes instances where mayn fule type for space heating is not "Gas"
        '''
        self.ds_obsrv_vars = self.ds_obsrv_vars[self.ds_obsrv_vars.Mainfueltype != 2]
        self.ds_obsrv_vars = self.ds_obsrv_vars[self.ds_obsrv_vars.Mainfueltype != 3]
        return
    
    
    def filter_for_income(self) -> None:
        '''
        Removes instances where household income is bigger than £100k (as they are lumped all together 
        above that value) or smaller than £1k
        '''
        self.ds_obsrv_vars = self.ds_obsrv_vars[self.ds_obsrv_vars.V_7 > 1000.]
        self.ds_obsrv_vars = self.ds_obsrv_vars[self.ds_obsrv_vars.V_7 < 99999.]
        return