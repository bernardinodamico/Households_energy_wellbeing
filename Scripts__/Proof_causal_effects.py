from CausalGraphicalModel import CausalGraphicalModel
import pyAgrum.causal as csl
from DataFusion import gen_training_dataset

'''
Script to:
- evaluate independency checks (using d-separartion) 
- count undirected simple backdoor paths between variables 

These information is used to perform the do-calculus computations carried out manually, 
so to obtain causal estimator formulae for the covariate-specific
causal effect P(Y_0 | do(X), W)).
'''

#set bin number for real-valued variables
Y0bn = 50
Wbn = 12
V1bn = 12
V7bn = 12
Laplace_sm = 0.001
ref_year = 2018 # the reference year for the dataset. Any of the following: 2015, 2016, 2017, 2018

# Generate training dataset 
discretised_dtset = gen_training_dataset(ref_year=ref_year, Y_0_bins_num=Y0bn, W_bins_num=Wbn, V_1_bins_num=V1bn, V_7_bins_num=V7bn)

# Initialise Causal Graphical Model
cg_model = CausalGraphicalModel(disctetised_ds=discretised_dtset)
cg_model.set_bin_numbers(Y_0_bins_num=Y0bn, W_bins_num=Wbn, V_1_bins_num=V1bn, V_7_bins_num=V7bn)
cg_model.set_Lp_smoothing(Lp_sm=Laplace_sm)
cg_model.build()

#extract subgraphs from the model:
G_full = cg_model.G
G_X_underscored = cg_model.get_subgraph(which_graph='G_X_underscored')
G_X_overscored = cg_model.get_subgraph(which_graph='G_X_overscored')

#checks for independencies in the graphs via d-separation
cg_model.check_independence(graph=G_X_underscored, A_nodes={'Y_0'}, B_nodes={'X'}, conditioned_on={'W', 'V_7'}, print_res=True)
cg_model.check_independence(graph=G_X_underscored, A_nodes={'V_7', 'W'}, B_nodes={'X'}, conditioned_on={'V_2'}, print_res=True)
cg_model.check_independence(graph=G_X_overscored, A_nodes={'V_2'}, B_nodes={'X'}, conditioned_on=set(), print_res=True)
cg_model.check_independence(graph=G_X_underscored, A_nodes={'W'}, B_nodes={'X'}, conditioned_on={'V_2'}, print_res=True)



cg_model.get_paths(sub_graph=G_X_underscored, st_var="Y_0", end_var="X", print_paths=False)
cg_model.get_paths(sub_graph=G_X_underscored, st_var="V_7", end_var="X", print_paths=False)
cg_model.get_paths(sub_graph=G_X_underscored, st_var="W", end_var="X", print_paths=False)
cg_model.get_paths(sub_graph=G_X_overscored, st_var="V_2", end_var="X", print_paths=False)


