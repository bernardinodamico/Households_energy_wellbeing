import pyAgrum as gum
from pandas import DataFrame
import pandas as pd
import pyAgrum.causal as csl
from CausalGraphicalModel import CausalGraphicalModel
from pyAgrum import Potential
from DataFusion import gen_training_dataset
from Estimator import Estimator, ComputeEffects
from Plotter import Plotter
from Values_mapping import GetVariableValues
pd.option_context('display.max_rows', None)


'''
#set bin number for real-valued variables
Y0bn = 35
Wbn = 13 
V1bn = 13
V7bn = 13
Laplace_sm = 0.001

discretised_dtset = gen_training_dataset(Y_0_bins_num=Y0bn, W_bins_num=Wbn, V_1_bins_num=V1bn, V_7_bins_num=V7bn)

ce = ComputeEffects()
p_Y0_given_doXx_1G, p_Y0_given_doXx_2G, exp_Y0_given_doXx_1G, exp_Y0_given_doXx_2G = ce.compute_ATE(Y_0_bins_num=Y0bn, W_bins_num=Wbn, V_1_bins_num=V1bn, V_7_bins_num=V7bn, Laplace_sm=Laplace_sm, dd=discretised_dtset)

# Plot ATEs 
plotter = Plotter()
plotter.plot_ATE(figure_name=f'ATE', 
                 width_cm=8., 
                 height_cm=10.,
                 doXx_1_distrib=p_Y0_given_doXx_1G, 
                 doXx_2_distrib=p_Y0_given_doXx_2G,
                 exp_Xx_1=exp_Y0_given_doXx_1G,
                 exp_Xx_2=exp_Y0_given_doXx_2G,
                 )

#===============================================================================================================
'''
#set bin number for real-valued variables
Y0bn = 10
Wbn = 12
V1bn = 13
V7bn = 13
Laplace_sm = 0.002

# Generate training dataset 
discretised_dtset = gen_training_dataset(Y_0_bins_num=Y0bn, W_bins_num=Wbn, V_1_bins_num=V1bn, V_7_bins_num=V7bn)

ce = ComputeEffects()
list_w, list_distribs_doXx_1, list_distribs_doXx_2, list_exp_Y0_given_doXx_1_Ww_1, list_exp_Y0_given_doXx_2_Ww_1 = ce.compute_CATE(Y_0_bins_num=Y0bn, W_bins_num=Wbn, V_1_bins_num=V1bn, V_7_bins_num=V7bn, Laplace_sm=Laplace_sm, dd=discretised_dtset)


# Plot CATEs
plotter = Plotter()
plotter.plot_CATE(figure_name=f'CATE',
                  width_cm=12.,
                  height_cm=11.,
                  w_values=list_w,
                  list_distribs_doXx_1=list_distribs_doXx_1,
                  list_exp_Y0_given_doXx_1_Ww_1=list_exp_Y0_given_doXx_1_Ww_1,
                  list_distribs_doXx_2=list_distribs_doXx_2,
                  list_exp_Y0_given_doXx_2_Ww_1=list_exp_Y0_given_doXx_2_Ww_1
                  )

