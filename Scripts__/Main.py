import pyAgrum as gum
from pandas import DataFrame
import pandas as pd
import pyAgrum.causal as csl
from CausalGraphicalModel import CausalGraphicalModel
from pyAgrum import Potential
from DataFusion import gen_training_dataset
from Estimator import Estimator
from Plotter import Plotter
pd.option_context('display.max_rows', None)



# Generate training dataset 
gen_training_dataset()


# Initialise Causal Graphical Model
cg_model = CausalGraphicalModel(dataset_filename='discretised_processed_dataset.csv')
cg_model.build()


# obtain causal effect distributions i.e. P(Y_0 | do(X=x)) and P(Y_0 | do(X=x), W=w)
est = Estimator(cg_model=cg_model)
p_Y0_given_doXx_1 = est.effect_distribution(X_val='1', use_label_vals=True)
p_Y0_given_doXx_2 = est.effect_distribution(X_val='2', use_label_vals=True)
p_Y0_given_doXx_Ww = est.cov_specific_effect_distribution(X_val='1', W_val='1', use_label_vals=False)

# obtain causal effect expectations i.e. E(Y_0 | do(X=x)) and E(Y_0 | do(X=x), W=w)
exp_Y0_given_doXx_1 = est.expectation(df_Xx=p_Y0_given_doXx_1, val_col_name='Y_0', prob_col_name='P(Y_0 | do(X=1))')
exp_Y0_given_doXx_2 = est.expectation(df_Xx=p_Y0_given_doXx_2, val_col_name='Y_0', prob_col_name='P(Y_0 | do(X=2))')


print(p_Y0_given_doXx_1)


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


'''
NOTE: re-run the gen_training_dataset() twice, the first time with 22 bins for Y_0 when computing the ATE. The second time with 10 bins or less when computing the CATE
Also conside twiking the Laplace smoothing if needed.
'''
