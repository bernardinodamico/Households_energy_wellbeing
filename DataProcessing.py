import pandas as pd
from pandas import DataFrame

class DataProcessing():

    

    def initialise_dset_obsrv_vars(self) -> DataFrame:
        '''
        Create ann empy DataFrame with variable Symbols as column headings 
        '''
        df = pd.DataFrame(columns=['X','Y_0','Y_1','W','V_0','V_1','V_2','V_3','V_4','V_5','V_6','V_7','V_8'])

        return df