import pyAgrum as gum
from pandas import DataFrame, Series
import pandas as pd
import pyAgrum.causal as csl
from CausalGraphicalModel import CausalGraphicalModel
from pyAgrum import Potential
from Values_mapping import GetVariableValues
from DataFusion import gen_training_dataset
pd.option_context('display.max_rows', None)



class Estimator():
    causal_grap_model: CausalGraphicalModel = None
    p_Y0_given_do_X_W: Potential = None
    p_Y0_given_do_X: Potential = None


    def __init__(self, cg_model: CausalGraphicalModel):
        self.causal_grap_model = cg_model
        return


    def cov_specific_causal_effect(self) -> DataFrame:
        '''
        Returns a pyAgrum Potential representing the full potential P(Y_0 | do(X), W)
        i.e. the distributions of Y_0 for all values of X=x and W=w
        '''
        ve = gum.VariableElimination(self.causal_grap_model.b_net)
        p_V7_W_given_X_V2 = ve.evidenceJointImpact(targets=['V_7', 'W'], evs={'X', 'V_2'}) #returns a pyAgrum.Potential for P(targets|evs) for all instantiations (values) of targets and evs variables. 
        p_W_given_X_V2 = ve.evidenceJointImpact(targets=['W'], evs={'X', 'V_2'})
        p_V2 = ve.evidenceJointImpact(targets=['V_2'], evs={})
        p_Y0_given_X_W_V7 = ve.evidenceJointImpact(targets=['Y_0'], evs={'X', 'W', 'V_7'})
        manual_estimate_do_X_given_W = (p_Y0_given_X_W_V7 * ((p_V7_W_given_X_V2 * p_V2).sumOut(['V_2']) / (p_W_given_X_V2 * p_V2).sumOut(['V_2']))).sumOut(['V_7'])
        self.p_Y0_given_do_X_W = manual_estimate_do_X_given_W

        return 
    

    def causal_effect(self) -> DataFrame:
        '''
        Returns a pyAgrum Potential representing the full potential P(Y_0 | do(X))
        i.e. the distributions of Y_0 for all values of X=x
        '''
        ve = gum.VariableElimination(self.causal_grap_model.b_net)
        p_W_given_X_V2 = ve.evidenceJointImpact(targets=['W'], evs={'X', 'V_2'})
        p_V2 = ve.evidenceJointImpact(targets=['V_2'], evs={})
        manual_estimate_do_X = (self.p_Y0_given_do_X_W * (p_W_given_X_V2 * p_V2).sumOut(['V_2'])).sumOut(['W'])

        self.p_Y0_given_do_X = manual_estimate_do_X 

        return 


    def cov_specific_effect_distribution(self, X_val: str, W_val: str, use_label_vals: bool = False) -> DataFrame:
        '''
        Inputs:
        - X_val: the assignment value for the interveening variable (either '1' or '2')
        - W_val: the assignment value for the observed covariate.
        - use_label_vals: defaul = False. If True, replaces the numerical values of Y_0 with the label values.

        Returns a two-column dataframe where the first column are Y_0 values and the second column 
        the corresponding (covariate_specific) post-intervention probabilities
        '''
        self.cov_specific_causal_effect()
        self.causal_effect()

        pot = self.p_Y0_given_do_X_W.extract({'X':X_val, 'W': W_val})
        pSeries: Series = pot.topandas()

        pSeries = pSeries.reset_index(level=0, drop=True)
        df = pSeries.to_frame().reset_index()
        df.columns = ['Y_0', f'P(Y_0 | do(X={X_val}), W={W_val})']

        if use_label_vals is True:
            label_list = GetVariableValues.get_labels(var_symbol='Y_0')
            for i in range(0, len(label_list)):
                df.loc[i, 'Y_0'] = label_list[i]
        p_Y0_given_doXx_Ww = df

        return p_Y0_given_doXx_Ww
    

    def effect_distribution(self, X_val: str, use_label_vals: bool = False) -> DataFrame:
        '''
        Inputs:
        - X_val: the assignment value for the interveening variable (either '1' or '2')
        - use_label_vals: defaul = False. If True, replaces the numerical values of Y_0 with the label values.

        Returns a two-column dataframe where the first column are Y_0 values and the second column 
        the corresponding post-intervention probabilities
        '''
        self.cov_specific_causal_effect()
        self.causal_effect()

        pot = self.p_Y0_given_do_X.extract({'X':X_val})
        pSeries: Series = pot.topandas()

        pSeries = pSeries.reset_index(level=0, drop=True)
        df = pSeries.to_frame().reset_index()
        df.columns = ['Y_0', f'P(Y_0 | do(X={X_val}))']

        if use_label_vals is True:
            label_list = GetVariableValues.get_labels(var_symbol='Y_0')
            for i in range(0, len(label_list)):
                df.loc[i, 'Y_0'] = label_list[i]
        p_Y0_given_doXx = df

        return p_Y0_given_doXx