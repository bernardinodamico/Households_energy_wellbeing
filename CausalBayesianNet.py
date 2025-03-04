import pyAgrum as gum
from pandas import DataFrame
import pandas as pd
import pyAgrum.causal as csl
from pyAgrum.causal import CausalModel
import networkx as nx
from networkx import DiGraph
from pyAgrum import BayesNet


class CausalBayesianNet():
    dataset_filename: str = None 
    b_net: BayesNet = None
    c_model: CausalModel = None
    G: DiGraph = None
 
    def __init__(self, dataset_filename: str):
        """
        parameter: dataset_filename = the name of the training dataset (including its the file extention)
        """
        self.dataset_filename = dataset_filename
        return

    def add_nodes(self) -> None:
        self.b_net = gum.BayesNet("MyCausalBN")

        self.b_net.add(gum.LabelizedVariable('X', "External wall insulation" , ['0', '1'])) 
        self.b_net.add(gum.LabelizedVariable('Y_0', "Heating energy use" , ['0', '1'])) 
        self.b_net.add(gum.LabelizedVariable('Y_1', "Indoor temperature" , ['0', '1']))
        self.b_net.add(gum.LabelizedVariable('W', "xxx" , ['0', '1']))
        self.b_net.add(gum.LabelizedVariable('V_0', "xxx" , ['0', '1']))
        self.b_net.add(gum.LabelizedVariable('V_1', "xxx" , ['0', '1']))
        self.b_net.add(gum.LabelizedVariable('V_2', "xxx" , ['0', '1']))
        self.b_net.add(gum.LabelizedVariable('V_3', "xxx" , ['0', '1']))
        self.b_net.add(gum.LabelizedVariable('V_4', "xxx" , ['0', '1']))
        self.b_net.add(gum.LabelizedVariable('V_5', "xxx" , ['0', '1']))
        self.b_net.add(gum.LabelizedVariable('V_6', "xxx" , ['0', '1']))
        self.b_net.add(gum.LabelizedVariable('V_7', "xxx" , ['0', '1']))
        self.b_net.add(gum.LabelizedVariable('V_8', "xxx" , ['0', '1']))
        return
    

    def add_causal_edges(self) -> None:
        self.b_net.addArc('X', 'Y_0') # X causes Y_1
        self.b_net.addArc('X', 'Y_1')
        self.b_net.addArc('Y_0', 'Y_1')
        self.b_net.addArc('W', 'Y_0')
        self.b_net.addArc('V_2', 'X')
        self.b_net.addArc('V_3', 'X')
        self.b_net.addArc('V_0', 'Y_1')
        self.b_net.addArc('V_0', 'Y_0')
        self.b_net.addArc('V_4', 'Y_0')
        self.b_net.addArc('V_4', 'V_1')
        self.b_net.addArc('V_5', 'V_4')
        self.b_net.addArc('V_6', 'V_1')
        self.b_net.addArc('V_6', 'Y_0')
        self.b_net.addArc('V_6', 'V_4')
        self.b_net.addArc('V_7', 'V_6')
        self.b_net.addArc('V_7', 'W')
        self.b_net.addArc('V_8', 'V_7')
        self.b_net.addArc('V_8', 'Y_0')
        self.b_net.addArc('V_8', 'V_1')
        self.b_net.addArc('V_7', 'V_2')
        self.b_net.addArc('V_1', 'W')
        self.b_net.addArc('V_0', 'V_1')
        self.b_net.addArc('X', 'V_1')
        return
    

    def get_subgraph(self, which_graph: str) -> DiGraph:
        self.define_graph()
        G_subgraph = self.G
        if which_graph == 'G_X_underscored': # the mutilated graph G with all arrows out of X being removed 
            G_subgraph.remove_edge('X', 'V_1')
            G_subgraph.remove_edge('X', 'Y_1')
            G_subgraph.remove_edge('X', 'Y_0')
        elif which_graph == 'G_X_overscored': # the mutilated graph G with all arrows into X being removed 
            G_subgraph.remove_edge('V_2', 'X')
            G_subgraph.remove_edge('V_3', 'X')
        elif which_graph == 'G': # the original graph G
            pass
        return G_subgraph


    def get_paths(self, sub_graph: DiGraph, st_var: str, end_var: str) -> list:
        """
        Returns the list of all undirected paths between two nodes. 
        Parameters: 
        - sub_graph: the graph to inspect
        - st_var: name of one of the variables
        - end_var: name of the other variable
        """
        print(f"\n All existing undirected paths from {st_var} to {end_var} in the provided graph:")
        paths = []
        for path in nx.all_simple_paths(G=sub_graph.to_undirected(), source=st_var, target=end_var, cutoff=None):
            print(path)
            paths.append(path)
        return paths


    def define_graph(self) -> None:
        self.G = nx.DiGraph()
        for (parent, child) in self.b_net.arcs():
            parent_name = self.b_net.variable(parent).name()  
            child_name = self.b_net.variable(child).name()
            self.G.add_edge(parent_name, child_name)
        return
    

    def learn_params(self, data_file_name: str) -> None:
        data = pd.read_csv(filepath_or_buffer=f"DATA/{data_file_name}")
        learner = gum.BNLearner(data, self.b_net)
        learner.useSmoothingPrior(1) # Laplace smoothing (e.g. a count C is replaced by C+1)
        self.b_net = learner.learnParameters(self.b_net.dag())
        return
    

    def add_latent_vars(self)->None:
        self.c_model = csl.CausalModel(bn=self.b_net, 
                                  latentVarsDescriptor=[("U_0", ["V_7","Y_0", "V_1"]),
                                                        ("U_1", ["V_6","V_7"]),
                                                        ("U_2", ["V_1"]),
                                                        ("U_3", ["V_1", "Y_0", "Y_1"]),
                                                        ("U_4", ["V_1", "Y_0", "Y_1"]),
                                                        ("U_5", ["Y_0", "V_1"]),
                                                        ],
                                  keepArcs=True)
        return
    

    def build(self) -> None:
        self.add_nodes()
        self.add_causal_edges()
        self.learn_params(data_file_name=self.dataset_filename)
        self.add_latent_vars()
        return

    


'''
===============================================================================================
this below works correctly. Implement it into a script to generate the proof for the hand do-calculus
i.e. generate all paths to figure out those blocked by colliders and those blocked by the adjustment set.
'''


cbn = CausalBayesianNet(dataset_filename='dataset_observed_variables.csv')
cbn.build()

estimand, estimate_do_X, message = csl.causalImpact(cm=cbn.c_model, on="Y_0", doing="X", knowing={"W"}, values={"X":'1'})


G_X_underscored = cbn.get_subgraph(which_graph='G_X_underscored')
cbn.get_paths(sub_graph=G_X_underscored,st_var='Y_0', end_var='X')

indp = cbn.b_net.isIndependent(['X'],['V_7'],['V_2']) #check if first variables are independent of second variables conditional on third variables
print(indp)