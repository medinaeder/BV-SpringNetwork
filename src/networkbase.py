import networkx as nx 
import numpy as np
import abc

class NetworkBase:
    """
    Class to represent Arbitrary Bolei-Vincent Network
    """
    __metaclass__ = abc.ABCMeta
    def __init__(self, G):
        self.preprocess(G)
    
    def preprocess(self, G):
        """
        Convert the graph, G, to use integer number
        Create a matrix of edge vectors e_R

        """

        self.number_of_nodes = G.number_of_nodes()

        G = nx.convert_node_labels_to_integers(G,label_attribute = "loc")

        # Change Edge Data Structure
        self.edges = np.array(G.edges())

        # Precompute Edge Vectors
        self.e_R = np.zeros((G.number_of_edges(), 2))
        for i,e in enumerate(G.edges()):
            (x1,y1) = G.nodes[e[0]]['loc']
            (x2,y2) = G.nodes[e[1]]['loc']
            self.e_R[i,0] = x2-x1
            self.e_R[i,1] = y2-y1

        # Store the Graph
        self.G = G

        # Intialize Solution
        self.y0 = np.zeros(self.number_of_nodes*6)
        non = self.number_of_nodes
        for i in range(non):
            self.y0[2*non+i] = 1e-6*(np.random.rand()-0.5)

        return 

    @abc.abstractmethod
    def internal_forces(self, u,v,tt):
        pass

    @abc.abstractmethod
    def external_forces(self,t):
        pass

    @abc.abstractmethod
    def damp_forces(self,ud,vd,ttd):
        pass

    @abc.abstractmethod
    def dynamicBCs(self, t, ud, vd, ttd):
        pass
    
    @abc.abstractmethod
    def staticBCs(self, t):
        pass
    
    @abc.abstractmethod
    def jacobian(self):
        pass

    # TODO: Finish the following implementation
    def KE(self,ud,vd,ttd):
        pass

    def PE(self,ud,vd,ttd):
        pass

    def jacobian(self):
        pass

