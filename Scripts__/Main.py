import pyAgrum as gum
from pandas import DataFrame
import pandas as pd
import pyAgrum.causal as csl
from CausalGraphicalModel import CausalGraphicalModel
from pyAgrum import Potential
from DataFusion import gen_training_dataset
from Estimator import Estimator
import matplotlib.pyplot as plt
pd.option_context('display.max_rows', None)



# Generate training dataset 
#gen_training_dataset()


# Initialise Causal Graphical Model
cg_model = CausalGraphicalModel(dataset_filename='discretised_processed_dataset.csv')
cg_model.build()


# obtain causal effect distributions i.e. P(Y_0 | do(X=x)) and P(Y_0 | do(X=x), W=w)
est = Estimator(cg_model=cg_model)
p_Y0_given_doXx_1 = est.effect_distribution(X_val='1', use_label_vals=True)
p_Y0_given_doXx_2 = est.effect_distribution(X_val='2', use_label_vals=True)
p_Y0_given_doXx_Ww = est.cov_specific_effect_distribution(X_val='1', W_val='1', use_label_vals=False)

print(p_Y0_given_doXx_1)








plt.bar(x=p_Y0_given_doXx_1['Y_0'], height=p_Y0_given_doXx_1['P(Y_0 | do(X=1))'], width=1, edgecolor='black', alpha=0.6, label="False")
plt.bar(x=p_Y0_given_doXx_2['Y_0'], height=p_Y0_given_doXx_2['P(Y_0 | do(X=2))'], width=1, edgecolor='black', alpha=0.6, label="True")

plt.rcParams["font.family"] = "Arial"
plt.xlabel(r'Gas consumtion $(Y_0)$ [kWh/year]', fontsize=12)
plt.ylabel(r'$P(Y_0 \mid do(X))$', fontsize=12)

plt.minorticks_on()
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.05))  # Major ticks every 0.01 units
plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.01))

plt.tick_params(axis='x', which='major', direction='out', length=6, labelsize=10)
plt.tick_params(axis='x', which='minor', direction='in', length=0)
plt.tick_params(axis='y', which='major', direction='in', length=6, labelsize=10)
plt.tick_params(axis='y', which='minor', direction='in', length=3)

plt.legend(title="Wall insulation (X)", frameon=False, fontsize=12, title_fontsize=12)

plt.xticks(rotation=90)
plt.tight_layout()
plt.ylim(0, 0.15)

plt.show()
