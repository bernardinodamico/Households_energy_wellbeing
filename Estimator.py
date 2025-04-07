import pyAgrum as gum
from pandas import DataFrame, Series
import pandas as pd
import pyAgrum.causal as csl
from CausalGraphicalModel import CausalGraphicalModel
from pyAgrum import Potential
from Values_mapping import GetVariableValues
import numpy as np
from DataFusion import gen_training_dataset
pd.option_context('display.max_rows', None)



class Estimator():
    causal_grap_model: CausalGraphicalModel = None
    p_Y0_given_do_X_W: Potential = None
    p_Y0_given_do_X: Potential = None

    Y0bn: int = None
    Wbn: int = None
    V1bn: int = None
    V7bn: int = None


    def __init__(self, cg_model: CausalGraphicalModel, Y_0_bins_num: int, W_bins_num: int, V_1_bins_num: int, V_7_bins_num: int):
        self.causal_grap_model = cg_model

        self.Y0bn = Y_0_bins_num
        self.Wbn = W_bins_num
        self.V1bn = V_1_bins_num
        self.V7bn = V_7_bins_num
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
        
        #estimand, manual_estimate_do_X_given_W , message = csl.causalImpact(cm=self.causal_grap_model.c_model, on="Y_0", doing="X", knowing={"W"})
        
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

        #estimand, manual_estimate_do_X , message = csl.causalImpact(cm=self.causal_grap_model.c_model, on="Y_0", doing="X")
        
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
            label_list = GetVariableValues.get_labels(var_symbol='Y_0', Y0bn=self.Y0bn, Wbn=self.Wbn, V1bn=self.V1bn, V7bn=self.V7bn)
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
            label_list = GetVariableValues.get_labels(var_symbol='Y_0', Y0bn=self.Y0bn, Wbn=self.Wbn, V1bn=self.V1bn, V7bn=self.V7bn)
            for i in range(0, len(label_list)):
                df.loc[i, 'Y_0'] = label_list[i]
        p_Y0_given_doXx = df

        return p_Y0_given_doXx
    

    def expectation(self, df_Xx: DataFrame, val_col_name: str, prob_col_name: str) -> float:
        '''
        Returns the expected value of a probability distribution. Inputs are:
        - df_Xx: a two-column dataframe reporting the post-intervention probability 
        distribution, e.g. P(Y_0 | do(X=1)
        - val_col_name: the name of the values column
        - prob_col_name: the name of the probabiity column
        '''
        df_Xx[val_col_name] = df_Xx[val_col_name].replace({'<': '', '>': '', ',': ''}, regex=True).astype(float)
        df_Xx[prob_col_name] = df_Xx[prob_col_name].astype(float)

        probabilities = df_Xx[prob_col_name].to_numpy()
        values = df_Xx[val_col_name].to_numpy()

        return np.sum(values * probabilities)






class ComputeEffects:

    @staticmethod
    def compute_ATEs(which: str, Y_0_bins_num: int, W_bins_num: int, V_1_bins_num: int, V_7_bins_num: int, Laplace_sm: float, dd: DataFrame) -> None:
        # Initialise Causal Graphical Model
        if which == 'ATE_G':
            cg_model = CausalGraphicalModel(disctetised_ds=dd, remove_W_Y0_edge=False)
        elif which == 'ATE_G_W_unders':
            cg_model = CausalGraphicalModel(disctetised_ds=dd, remove_W_Y0_edge=True)
        cg_model.set_bin_numbers(Y_0_bins_num=Y_0_bins_num, W_bins_num=W_bins_num, V_1_bins_num=V_1_bins_num, V_7_bins_num=V_7_bins_num)
        cg_model.set_Lp_smoothing(Lp_sm=Laplace_sm)
        cg_model.build()

        # obtain causal effect distributions i.e. P(Y_0 | do(X=x)) 
        est = Estimator(cg_model=cg_model, Y_0_bins_num=Y_0_bins_num, W_bins_num=W_bins_num, V_1_bins_num=V_1_bins_num, V_7_bins_num=V_7_bins_num)
        p_Y0_given_doXx_1 = est.effect_distribution(X_val='1', use_label_vals=True)
        p_Y0_given_doXx_2 = est.effect_distribution(X_val='2', use_label_vals=True)

        # obtain causal effect expectations i.e. E(Y_0 | do(X=x)) 
        exp_Y0_given_doXx_1 = est.expectation(df_Xx=p_Y0_given_doXx_1, val_col_name='Y_0', prob_col_name='P(Y_0 | do(X=1))')
        exp_Y0_given_doXx_2 = est.expectation(df_Xx=p_Y0_given_doXx_2, val_col_name='Y_0', prob_col_name='P(Y_0 | do(X=2))')

        return p_Y0_given_doXx_1, p_Y0_given_doXx_2, exp_Y0_given_doXx_1, exp_Y0_given_doXx_2