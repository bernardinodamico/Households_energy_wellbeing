import pyAgrum as gum
import pandas as pd
import pyAgrum.causal as csl
from pyAgrum.causal import CausalModel
import networkx as nx
from networkx import DiGraph
from pyAgrum import BayesNet
from Values_mapping import GetVariableValues as vvalues



class CausalGraphicalModel():
    dataset_filename: str = None 
    b_net: BayesNet = None
    c_model: CausalModel = None
    G: DiGraph = None
 
    def __init__(self, dataset_filename: str):
        """
        parameter: dataset_filename = the name of the training dataset (including its file extention)
        """
        self.dataset_filename = dataset_filename
        return

    def add_nodes(self) -> None:
        self.b_net = gum.BayesNet("MyCausalBN")

        self.b_net.add(gum.LabelizedVariable('X', "External walls insulation" , vvalues.get_nums('X'))) 
        self.b_net.add(gum.LabelizedVariable('Y_0', "Energy (gas) consumption" , vvalues.get_nums('Y_0'))) 
        self.b_net.add(gum.LabelizedVariable('W', "Energy burden" , vvalues.get_nums('W')))
        self.b_net.add(gum.LabelizedVariable('V_0', "Dwelling type" , vvalues.get_nums('V_0')))
        self.b_net.add(gum.LabelizedVariable('V_1', "Energy (gas) cost" , vvalues.get_nums('V_1')))
        self.b_net.add(gum.LabelizedVariable('V_2', "Tenancy" , vvalues.get_nums('V_2')))
        self.b_net.add(gum.LabelizedVariable('V_3', "Dwelling age" , vvalues.get_nums('V_3')))
        self.b_net.add(gum.LabelizedVariable('V_4', "Under Occupancy" , vvalues.get_nums('V_4')))
        self.b_net.add(gum.LabelizedVariable('V_5', "Household size" , vvalues.get_nums('V_5')))
        self.b_net.add(gum.LabelizedVariable('V_6', "Dwelling floor area" , vvalues.get_nums('V_6')))
        self.b_net.add(gum.LabelizedVariable('V_7', "Household income" , vvalues.get_nums('V_7')))
        self.b_net.add(gum.LabelizedVariable('V_8', "Household composition" , vvalues.get_nums('V_8')))

        return
    

    def add_causal_edges(self) -> None:
        self.b_net.addArc('X', 'Y_0') # X causes Y_0
        self.b_net.addArc('W', 'Y_0')
        self.b_net.addArc('V_2', 'X')
        self.b_net.addArc('V_3', 'X')
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
            G_subgraph.remove_edge('X', 'Y_0')
            G_subgraph.name = 'G_X_underscored'
        elif which_graph == 'G_X_overscored': # the mutilated graph G with all arrows into X being removed 
            G_subgraph.remove_edge('V_2', 'X')
            G_subgraph.remove_edge('V_3', 'X')
            G_subgraph.name = 'G_X_overscored'
        elif which_graph == 'G': # the original graph G
            pass
        else:
            raise ValueError('wrong value assigned to <which_graph> parameter ')
        return G_subgraph


    def get_paths(self, sub_graph: DiGraph, st_var: str, end_var: str, print_paths: bool = True, print_count: bool = True) -> list:
        """
        Returns the list (or count) of all undirected paths between two nodes. 
        Parameters: 
        - sub_graph: the graph to inspect
        - st_var: name of one of the variables
        - end_var: name of the other variable
        """
        paths = []
        for path in nx.all_simple_paths(G=sub_graph.to_undirected(), source=st_var, target=end_var, cutoff=None):
            paths.append(path)
        
        if print_count is True:
            print("\n")
            print(f"Found {len(paths)} undirected paths between {st_var} and {end_var} in graph {sub_graph.name}")
        if print_paths is True:
            for path in paths:
                print(path)
        return paths


    def define_graph(self) -> None:
        '''Add directed adges between observed variables to the DiGraph "G".'''
        self.G = nx.DiGraph()
        self.G.name = 'G'
        for (parent, child) in self.b_net.arcs():
            parent_name = self.b_net.variable(parent).name()  
            child_name = self.b_net.variable(child).name()
            self.G.add_edge(parent_name, child_name)
        
        '''Add directed adges from latent variables to the DiGraph "G".'''
        self.G.add_edge("U_0", "V_7")
        self.G.add_edge("U_0", "Y_0")
        self.G.add_edge("U_0", "V_1")
        self.G.add_edge("U_1", "V_6")
        self.G.add_edge("U_1", "V_7")
        self.G.add_edge("U_2", "V_1")
        self.G.add_edge("U_3", "V_1")
        self.G.add_edge("U_3", "Y_0")
        self.G.add_edge("U_4", "V_1")
        self.G.add_edge("U_4", "Y_0")
        self.G.add_edge("U_5", "Y_0")
        self.G.add_edge("U_5", "V_1")
        return
    

    def learn_params(self, data_file_name: str) -> None:
        data = pd.read_csv(filepath_or_buffer=f"DATA/{data_file_name}")
        learner = gum.BNLearner(data, self.b_net)
        learner.useSmoothingPrior(0.00001) # Laplace smoothing (e.g. a count C is replaced by C+1)
        self.b_net = learner.learnParameters(self.b_net.dag())
        return
    

    def add_latent_vars(self)-> None:
        '''
        Method to add latent variables to the PyAgrum CausalModel object "c_model". 
        '''
        self.c_model = csl.CausalModel(bn=self.b_net, 
                                  latentVarsDescriptor=[("U_0", ["V_7","Y_0", "V_1"]),
                                                        ("U_1", ["V_6","V_7"]),
                                                        ("U_2", ["V_1"]),
                                                        ("U_3", ["V_1", "Y_0"]),
                                                        ("U_4", ["V_1", "Y_0"]),
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
    

    def check_independence(self, graph: DiGraph, A_nodes: set, B_nodes: set, conditioned_on: set, print_res: bool = True) -> bool:
        '''
        Uses the d-separation algo to check for conditional (or absolute) independence between sets of nodes. Params:
        - graph: a networkx DiGraph object
        - A_nodes: a python-set of variables
        - B_nodes: another set of variables, so that independency is checked between the two sets
        - conditioned_on: the set of known variables. For absolute independency checks, just give an empy set as argument, i.e. conditioned_on=set()
        '''
        cond_indep = nx.is_d_separator(graph, A_nodes, B_nodes, conditioned_on)
        if print_res is True:
            print("\n")
            if len(conditioned_on) == 0:
                print(f"{A_nodes} _|_ {B_nodes} in graph {graph.name} it is: {cond_indep}")
            else:
                print(f"{A_nodes} _|_ {B_nodes} | {conditioned_on} in graph {graph.name} it is: {cond_indep}")
        return cond_indep

    



