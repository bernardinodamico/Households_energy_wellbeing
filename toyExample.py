import pyAgrum as gum
import pandas as pd
import pyAgrum.causal as csl

'''
1) Setting up of a Causal Bayesian Network
'''

# load dataset
data = pd.read_csv(filepath_or_buffer="DATA/frequency_data_toy_exampe.csv")

# instantiate a (Causal) Bayesian Network
bn = gum.BayesNet("MyCausalBN")

# add nodes to the (Causal) BN

id_X = bn.add(gum.LabelizedVariable('X', "the intervention variable" , ['True', 'False'])) 
id_Y = bn.add(gum.LabelizedVariable('Y', "the outcome variable" , ['True', 'False'])) 
id_Z = bn.add(gum.LabelizedVariable('Z', "a confounder" , ['True', 'False'])) 

#defines edges
bn.addArc('X', 'Y') # X causes Y
bn.addArc('Z', 'X')
bn.addArc('Z', 'Y')


# learn the parameters (i.e. the CPTs) of the BN
learner = gum.BNLearner(data, bn)
learner.useSmoothingPrior(1) # Laplace smoothing (e.g. a count C is replaced by C+1)
bn = learner.learnParameters(bn.dag())
#----------------------------------------------------------------------------------------


'''
2) Now that we have a Bayesian (causal) net, we can perform causal effect estimation.
For this model, the causal effect is estimable via back-door. We apply the back-door formula 
manually, hence obtaine the estimate via the below line of code, and print out the 
post-intervetional distribution P(Y | do(X)) for both values of X=0 and X=1.
'''
estimate_manual = (bn.cpt('Y') * bn.cpt('Z')).sumOut(['Z'])
print("    ")
print("_______________________________________________")
print("manually estimated P(Y | do(X)):")
print(estimate_manual)

'''
3) Now let repeat the same using PyAgrum. First we obtain the estimand (back-door formula)
and print it in Latex format. Then we print the resulting post-intervetional distribution 
(estimates) of Y for P(Y | do(X=0)) and P(Y | do(X=1))
'''

d = csl.CausalModel(bn=bn, latentVarsDescriptor=[("lat", ["X","Z"])])
estimand, estimate_do_X0, message = csl.causalImpact(cm=d, on="Y", doing="X", values={"X":'True'})


print("_______________________________________________")
print(f"PyAgrum estimand formula (in latex format): {estimand.toLatex()}")
print("_______________________________________________")
print("Pyagrum estimated P(Y | do(X=True)):")
print(estimate_do_X0)

estimand, estimate_do_X1, message = csl.causalImpact(cm=d, on="Y", doing="X", values={"X":'False'})

print("_______________________________________________")
print("PyAgrum estimated P(Y | do(X=False)):")
print(estimate_do_X1)


'''
NOTE: reference to PyAgrum documentation: https://pyagrum.readthedocs.io/en/1.17.2/notebooks/64-Causality_DoCalculusExamples.html?fbclid=IwY2xjawIQEtlleHRuA2FlbQIxMAABHSszzptUYcAYv0mmqJIa23GKpAiVjFgxYZAyWJTSkyU4DGBscDAhjwmHzw_aem_aH5lgf8fxh8BbXLMK0wemQ
'''