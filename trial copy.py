import pyAgrum as gum
from pandas import DataFrame
import pyAgrum.causal as csl
from CausalGraphicalModel import CausalGraphicalModel
from pyAgrum import Potential
from Scripts__.Build_training_dataset import gen_training_dataset



# Generate training dataset 
gen_training_dataset()


# Initialise Causal Graphical Model
cbn = CausalGraphicalModel(dataset_filename='discretised_processed_dataset.csv')
cbn.build()


'''
estimand, estimate_do_X, message = csl.causalImpact(cm=cbn.c_model, on="Y_0", doing="X", knowing={"W"}, values={"X":'1'})

print(f"message:{message}")
print("_______________________________________________")
print(f"PyAgrum estimator formula (in latex format): {estimand.toLatex()}")

print("_______________________________________________" )
print("PyAgrum estimate for P(Y_0 | do(X=0), W):")
print(estimate_do_X)
'''

ve = gum.VariableElimination(cbn.b_net)
p_V7_W_given_X_V2 = ve.evidenceJointImpact(targets=['V_7', 'W'], evs={'X', 'V_2'}) #returns a pyAgrum.Potential for P(targets|evs) for all instantiations (values) of targets and evs variables. 
p_W_given_X_V2 = ve.evidenceJointImpact(targets=['W'], evs={'X', 'V_2'})
p_V2 = ve.evidenceJointImpact(targets=['V_2'], evs={})
p_Y0_given_X_W_V7 = ve.evidenceJointImpact(targets=['Y_0'], evs={'X', 'W', 'V_7'})


manual_estimate_do_X_given_W = (p_Y0_given_X_W_V7 * ((p_V7_W_given_X_V2 * p_V2).sumOut(['V_2']) / (p_W_given_X_V2 * p_V2).sumOut(['V_2']))).sumOut(['V_7'])


print("_______________________________________________")
print("Hand-calculated estimate for P(Y_0 | do(X=1), W) and for P(Y_0 | do(X=2), W):")
print(manual_estimate_do_X_given_W)

manual_estimate_do_X_given_W: Potential
out = manual_estimate_do_X_given_W.topandas()
out.to_csv(path_or_buf='output.csv')


'''
extract the individual prob distributions form this potential above, e.g. P(Y_0 | do(X=1), W=w). These inividual distributions
are then processed to ontain ATE, CATE distributions...
'''
