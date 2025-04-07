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

p_Y0_given_doXx_1G, p_Y0_given_doXx_2G, exp_Y0_given_doXx_1G, exp_Y0_given_doXx_2G = ComputeEffects.compute_ATEs(which='ATE_G', Y_0_bins_num=Y0bn, W_bins_num=Wbn, V_1_bins_num=V1bn, V_7_bins_num=V7bn, Laplace_sm=Laplace_sm, dd=discretised_dtset)
p_Y0_given_doXx_1GW, p_Y0_given_doXx_2GW, exp_Y0_given_doXx_1GW, exp_Y0_given_doXx_2GW = ComputeEffects.compute_ATEs(which='ATE_G_W_unders', Y_0_bins_num=Y0bn, W_bins_num=Wbn, V_1_bins_num=V1bn, V_7_bins_num=V7bn, Laplace_sm=Laplace_sm, dd=discretised_dtset)

# Plot ATEs 
plotter = Plotter()
plotter.plot_ATEs(figure_name=f'ATE', 
                 width_cm=8., 
                 height_cm=11.,
                 doXx_1_distrib_G=p_Y0_given_doXx_1G, 
                 doXx_2_distrib_G=p_Y0_given_doXx_2G,
                 exp_Xx_1_G=exp_Y0_given_doXx_1G,
                 exp_Xx_2_G=exp_Y0_given_doXx_2G,
                 doXx_1_distrib_GW=p_Y0_given_doXx_1GW, 
                 doXx_2_distrib_GW=p_Y0_given_doXx_2GW,
                 exp_Xx_1_GW=exp_Y0_given_doXx_1GW,
                 exp_Xx_2_GW=exp_Y0_given_doXx_2GW,
                 )


'''
#===============================================================================================================

#set bin number for real-valued variables
Y0bn = 13
Wbn = 8
V1bn = 13
V7bn = 13
Laplace_sm = 0.002

# Generate training dataset 
discretised_dtset = gen_training_dataset(Y_0_bins_num=Y0bn, W_bins_num=Wbn, V_1_bins_num=V1bn, V_7_bins_num=V7bn)

# Initialise Causal Graphical Model
cg_model = CausalGraphicalModel(disctetised_ds=discretised_dtset, remove_W_Y0_edge=True) 
cg_model.set_bin_numbers(Y_0_bins_num=Y0bn, W_bins_num=Wbn, V_1_bins_num=V1bn, V_7_bins_num=V7bn)
cg_model.set_Lp_smoothing(Lp_sm=Laplace_sm)
cg_model.build()


est = Estimator(cg_model=cg_model, Y_0_bins_num=Y0bn, W_bins_num=Wbn, V_1_bins_num=V1bn, V_7_bins_num=V7bn)
for w in range(1, Wbn+1):
    # obtain causal effect distributions i.e. P(Y_0 | do(X=x), W=w)
    p_Y0_given_doXx_1_Ww_1 = est.cov_specific_effect_distribution(X_val='1', W_val=f'{w}', use_label_vals=True)
    p_Y0_given_doXx_2_Ww_1 = est.cov_specific_effect_distribution(X_val='2', W_val=f'{w}', use_label_vals=True)

    # obtain causal effect expectations i.e. E(Y_0 | do(X=x), W=w) 
    exp_Y0_given_doXx_1_Ww_1 = est.expectation(df_Xx=p_Y0_given_doXx_1_Ww_1, val_col_name='Y_0', prob_col_name=f'P(Y_0 | do(X=1), W={w})')
    exp_Y0_given_doXx_2_Ww_1 = est.expectation(df_Xx=p_Y0_given_doXx_2_Ww_1, val_col_name='Y_0', prob_col_name=f'P(Y_0 | do(X=2), W={w})')

    print(GetVariableValues.get_labels(var_symbol='W', Y0bn=Y0bn, Wbn=Wbn, V1bn=V1bn, V7bn=V7bn)[w-1], exp_Y0_given_doXx_2_Ww_1 - exp_Y0_given_doXx_1_Ww_1)


