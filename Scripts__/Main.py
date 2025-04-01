import pyAgrum as gum
from pandas import DataFrame
import pandas as pd
import pyAgrum.causal as csl
from CausalGraphicalModel import CausalGraphicalModel
from pyAgrum import Potential
from DataFusion import gen_training_dataset
from Estimator import Estimator
from Plotter import Plotter
from Values_mapping import GetVariableValues
pd.option_context('display.max_rows', None)


#set bin number for real-valued variables
Y0bn = 35
Wbn = 12
V1bn = 12
V7bn = 12
Laplace_sm = 0.001
# Generate training dataset 
gen_training_dataset(Y_0_bins_num=Y0bn, W_bins_num=Wbn, V_1_bins_num=V1bn, V_7_bins_num=V7bn)

# Initialise Causal Graphical Model
cg_model = CausalGraphicalModel(dataset_filename='discretised_processed_dataset.csv')
cg_model.set_bin_numbers(Y_0_bins_num=Y0bn, W_bins_num=Wbn, V_1_bins_num=V1bn, V_7_bins_num=V7bn)
cg_model.set_Lp_smoothing(Lp_sm=Laplace_sm)
cg_model.build()

# obtain causal effect distributions i.e. P(Y_0 | do(X=x)) 
est = Estimator(cg_model=cg_model, Y_0_bins_num=Y0bn, W_bins_num=Wbn, V_1_bins_num=V1bn, V_7_bins_num=V7bn)
p_Y0_given_doXx_1 = est.effect_distribution(X_val='1', use_label_vals=True)
p_Y0_given_doXx_2 = est.effect_distribution(X_val='2', use_label_vals=True)

# obtain causal effect expectations i.e. E(Y_0 | do(X=x)) 
exp_Y0_given_doXx_1 = est.expectation(df_Xx=p_Y0_given_doXx_1, val_col_name='Y_0', prob_col_name='P(Y_0 | do(X=1))')
exp_Y0_given_doXx_2 = est.expectation(df_Xx=p_Y0_given_doXx_2, val_col_name='Y_0', prob_col_name='P(Y_0 | do(X=2))')


# Plot ATE 
plotter = Plotter()
plotter.plot_ATE(figure_name='ATE', 
                 width_cm=8., 
                 height_cm=8.,
                 doXx_1_distrib=p_Y0_given_doXx_1, 
                 doXx_2_distrib=p_Y0_given_doXx_2,
                 exp_Xx_1=exp_Y0_given_doXx_1,
                 exp_Xx_2=exp_Y0_given_doXx_2
                 )



#================================================================================================================
'''
#set bin number for real-valued variables
Y0bn = 10
Wbn = 20
V1bn = 10
V7bn = 10
Laplace_sm = 0.01
# Generate training dataset 
gen_training_dataset(Y_0_bins_num=Y0bn, W_bins_num=Wbn, V_1_bins_num=V1bn, V_7_bins_num=V7bn)

# Initialise Causal Graphical Model
cg_model = CausalGraphicalModel(dataset_filename='discretised_processed_dataset.csv')
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

    print(GetVariableValues.get_labels(var_symbol='W', Y0bn=Y0bn, Wbn=Wbn, V1bn=V1bn, V7bn=V7bn)[w-1], exp_Y0_given_doXx_1_Ww_1 - exp_Y0_given_doXx_2_Ww_1)
'''


