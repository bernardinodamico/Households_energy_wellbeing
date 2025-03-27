import pyAgrum as gum
from pandas import DataFrame
import pandas as pd
import pyAgrum.causal as csl
from CausalGraphicalModel import CausalGraphicalModel
from pyAgrum import Potential
from DataFusion import gen_training_dataset
from Estimator import Estimator
pd.option_context('display.max_rows', None)



# Generate training dataset 
#gen_training_dataset()


# Initialise Causal Graphical Model
cg_model = CausalGraphicalModel(dataset_filename='discretised_processed_dataset.csv')
cg_model.build()


# obtain causal effect distributions i.e. P(Y_0 | do(X=x)) and P(Y_0 | do(X=x), W=w)
est = Estimator(cg_model=cg_model)
p_Y0_given_doXx = est.effect_distribution(X_val='1', use_label_vals=False)
p_Y0_given_doXx_Ww = est.cov_specific_effect_distribution(X_val='1', W_val='1', use_label_vals=False)

print(p_Y0_given_doXx)

'''
extract the individual prob distributions form this potential above, e.g. P(Y_0 | do(X=1), W=w). These inividual distributions
are then processed to onbtain ATE, CATE distributions...
'''
