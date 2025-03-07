import pyAgrum as gum
import pandas as pd
import pyAgrum.causal as csl
import networkx as nx


'''
1) Setting up of a Causal Bayesian Network
'''

# load dataset
data = pd.read_csv(filepath_or_buffer="DATA/dataset_observed_variables.csv")

# instantiate a (Causal) Bayesian Network
bn = gum.BayesNet("MyCausalBN")

# add nodes to the (Causal) BN
id_X = bn.add(gum.LabelizedVariable('X', "External wall insulation" , ['0', '1'])) 
id_Y_0 = bn.add(gum.LabelizedVariable('Y_0', "Heating energy use" , ['0', '1'])) 
id_Y_1 = bn.add(gum.LabelizedVariable('Y_1', "Indoor temperature" , ['0', '1']))
id_W = bn.add(gum.LabelizedVariable('W', "xxx" , ['0', '1']))
id_V_0 = bn.add(gum.LabelizedVariable('V_0', "xxx" , ['0', '1']))
id_V_1 = bn.add(gum.LabelizedVariable('V_1', "xxx" , ['0', '1']))
id_V_2 = bn.add(gum.LabelizedVariable('V_2', "xxx" , ['0', '1']))
id_V_3 = bn.add(gum.LabelizedVariable('V_3', "xxx" , ['0', '1']))
id_V_4 = bn.add(gum.LabelizedVariable('V_4', "xxx" , ['0', '1']))
id_V_5 = bn.add(gum.LabelizedVariable('V_5', "xxx" , ['0', '1']))
id_V_6 = bn.add(gum.LabelizedVariable('V_6', "xxx" , ['0', '1']))
id_V_7 = bn.add(gum.LabelizedVariable('V_7', "xxx" , ['0', '1']))
id_V_8 = bn.add(gum.LabelizedVariable('V_8', "xxx" , ['0', '1']))


#defines edges

bn.addArc('X', 'Y_0') # X causes Y_1
bn.addArc('X', 'Y_1')
bn.addArc('Y_0', 'Y_1')
bn.addArc('W', 'Y_0')
bn.addArc('V_2', 'X')
bn.addArc('V_3', 'X')
bn.addArc('V_0', 'Y_1')
bn.addArc('V_0', 'Y_0')
bn.addArc('V_4', 'Y_0')
bn.addArc('V_4', 'V_1')
bn.addArc('V_5', 'V_4')
bn.addArc('V_6', 'V_1')
bn.addArc('V_6', 'Y_0')
bn.addArc('V_6', 'V_4')
bn.addArc('V_7', 'V_6')
bn.addArc('V_7', 'W')
bn.addArc('V_8', 'V_7')
bn.addArc('V_8', 'Y_0')
bn.addArc('V_8', 'V_1')
bn.addArc('V_7', 'V_2')
bn.addArc('V_1', 'W')
bn.addArc('V_0', 'V_1')
bn.addArc('X', 'V_1')


# learn the parameters (i.e. the CPTs) of the BN
learner = gum.BNLearner(data, bn)
learner.useSmoothingPrior(1) # Laplace smoothing (e.g. a count C is replaced by C+1)
bn = learner.learnParameters(bn.dag())
#----------------------------------------------------------------------------------------


# identify causal effect (if possible at all)

d = csl.CausalModel(bn=bn, latentVarsDescriptor=[("U_0", ["V_7","Y_0", "V_1"]),
                                                 ("U_1", ["V_6","V_7"]),
                                                 ("U_2", ["V_1"]),
                                                 ("U_3", ["V_1", "Y_0", "Y_1"]),
                                                 ("U_4", ["V_1", "Y_0", "Y_1"]),
                                                 ("U_5", ["Y_0", "V_1"]),
                                                 ],
                                                 keepArcs=True
                                                 )


estimand, estimate_do_X, message = csl.causalImpact(cm=d, on="Y_1", doing="X", knowing={"W"}, values={"X":'0'})

print(f"message:{message}")
print("_______________________________________________")
print(f"PyAgrum estimator formula (in latex format): {estimand.toLatex()}")

print("_______________________________________________")
print("PyAgrum estimate for P(Y_1 | do(X=0), W):")
print(estimate_do_X)

'''
NOTE: sooo... the W-specific causal effect of X on Y_0 as well as Y_1 is identifiable according to PyAgrum. 
The resulting estimand formula is really cumbersome and involves a lot of variables...
I derived the estimand for this W-specific causal effect by hand using do-calculus, hence printed below: "manual_estimate_do_X"
the difference between the two estimeted causal effects is at most 0.1% for some values. Conversely, when computing the observational 
distribution P(Y_0 | X, W) its values differ by as much as 10% compared to the causal estimate(s)

'''


ve = gum.VariableElimination(bn)
p_V7_W_given_X_V2 = ve.evidenceJointImpact(targets=['V_7', 'W'], evs={'X', 'V_2'}) #returns a pyAgrum.Potential for P(targets|evs) for all instantiations (values) of targets and evs variables. 
p_W_given_X_V2 = ve.evidenceJointImpact(targets=['W'], evs={'X', 'V_2'})
p_V2 = ve.evidenceJointImpact(targets=['V_2'], evs={})
p_Y1_given_X_W_V7 = ve.evidenceJointImpact(targets=['Y_1'], evs={'X', 'W', 'V_7'})


manual_estimate_do_X = (p_Y1_given_X_W_V7 * ((p_V7_W_given_X_V2 * p_V2).sumOut(['V_2']) / (p_W_given_X_V2 * p_V2).sumOut(['V_2']))).sumOut(['V_7'])


print("_______________________________________________")
print("Hand-calculated estimate for P(Y_1 | do(X=0), W) and for P(Y_1 | do(X=1), W):")
print(manual_estimate_do_X)



