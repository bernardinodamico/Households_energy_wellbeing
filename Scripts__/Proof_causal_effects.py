from CausalGraphicalModel import CausalGraphicalModel


'''
Code to evaluate independency checks underpinning the Do-calculus computations
carried out manually, so to obtain causal estimand formulae for the covariate-specific
causal effects P(Y_0 | do(X), W)) and P(Y_1 | do(X), W)).
'''
#load training dataset:
cbn = CausalGraphicalModel(dataset_filename='dataset_observed_variables.csv')
cbn.build()

#extract subgraphs from the model:
G_full = cbn.G
G_X_underscored = cbn.get_subgraph(which_graph='G_X_underscored')
G_X_overscored = cbn.get_subgraph(which_graph='G_X_overscored')

#check for independecies in the graphs via d-separation
cbn.check_independence(graph=G_X_underscored, A_nodes={'Y_0'}, B_nodes={'X'}, conditioned_on={'W', 'V_7'}, print_res=True)
cbn.check_independence(graph=G_X_underscored, A_nodes={'V_7', 'W'}, B_nodes={'X'}, conditioned_on={'V_2'}, print_res=True)
cbn.check_independence(graph=G_X_overscored, A_nodes={'V_2'}, B_nodes={'X'}, conditioned_on=set(), print_res=True)
cbn.check_independence(graph=G_X_underscored, A_nodes={'W'}, B_nodes={'X'}, conditioned_on={'V_2'}, print_res=True)



'''
'''
# Use the below to make causal estimates to compare with the manually calculated ones - so to make a further "sanity check" 
#estimand, estimate_do_X, message = csl.causalImpact(cm=cbn.c_model, on="Y_0", doing="X", knowing={"W"}, values={"X":'1'})