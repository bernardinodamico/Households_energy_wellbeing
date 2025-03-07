from CausalGraphicalModel import CausalGraphicalModel
import pyAgrum.causal as csl

'''
Script to:
- evaluate independency checks (using d-separartion) 
- count undirected simple backdoor paths between variables 

These information is used to perform the do-calculus computations carried out manually, 
so to obtain causal estimator formulae for the covariate-specific
causal effects P(Y_0 | do(X), W)) and P(Y_1 | do(X), W)).
'''

#load training dataset:
cbn = CausalGraphicalModel(dataset_filename='dataset_observed_variables.csv')
cbn.build()

#extract subgraphs from the model:
G_full = cbn.G
G_X_underscored = cbn.get_subgraph(which_graph='G_X_underscored')
G_X_overscored = cbn.get_subgraph(which_graph='G_X_overscored')

#checks for independencies in the graphs via d-separation
cbn.check_independence(graph=G_X_underscored, A_nodes={'Y_0'}, B_nodes={'X'}, conditioned_on={'W', 'V_7'}, print_res=True)
cbn.check_independence(graph=G_X_underscored, A_nodes={'V_7', 'W'}, B_nodes={'X'}, conditioned_on={'V_2'}, print_res=True)
cbn.check_independence(graph=G_X_overscored, A_nodes={'V_2'}, B_nodes={'X'}, conditioned_on=set(), print_res=True)
cbn.check_independence(graph=G_X_underscored, A_nodes={'W'}, B_nodes={'X'}, conditioned_on={'V_2'}, print_res=True)

cbn.check_independence(graph=G_X_underscored, A_nodes={'Y_1'}, B_nodes={'X'}, conditioned_on={'W', 'V_7'}, print_res=True)



cbn.get_paths(sub_graph=G_X_underscored, st_var="Y_0", end_var="X", print_paths=False)
cbn.get_paths(sub_graph=G_X_underscored, st_var="V_7", end_var="X", print_paths=False)
cbn.get_paths(sub_graph=G_X_underscored, st_var="W", end_var="X", print_paths=False)
cbn.get_paths(sub_graph=G_X_overscored, st_var="V_2", end_var="X", print_paths=False)

cbn.get_paths(sub_graph=G_X_underscored, st_var="Y_1", end_var="X", print_paths=False)

