import pyAgrum as gum
import pandas as pd



from IPython.display import display, Math, Latex, HTML
import pyAgrum.causal as csl

# load dataset
data = pd.read_csv(filepath_or_buffer="DATA/prova_frequency_data.csv")

# instantiate a Causal Bayesian Network
bn = gum.BayesNet("MyCausalBN")

# add nodes to the Causal BN
id_X = bn.add(gum.LabelizedVariable('X', "the intervention variable" , 2)) 
id_Y = bn.add(gum.LabelizedVariable('Y', "the outcome variable" , 2)) 
id_Z = bn.add(gum.LabelizedVariable('Z', "a confounder" , 2)) 


#defines edges
bn.addArc('X', 'Y') # X causes Y
bn.addArc('Z', 'X')
bn.addArc('Z', 'Y')



# learn the parameters (i.e. the CPTs)
learner = gum.BNLearner(data, bn)
#learner.useSmoothingPrior(1000) # Laplace smoothing (e.g. a count C is replaced by C+1000)
bn = learner.learnParameters(bn.dag())
#print(bn.cpt(id_Z))



d = csl.CausalModel(bn=bn, latentVarsDescriptor=[("lat", ["X","Z"])])
res = csl.causalImpact(cm=d, on="Y", doing="X", values={"X":0})


estimate = res[1]
estimand = res[0]

print("__________")
print(estimand.toLatex())
print(estimate)

#----------------------------------------


estimate_manual = (bn.cpt('Y') * bn.cpt('Z')).sumOut(['Z'])
print(estimate_manual)