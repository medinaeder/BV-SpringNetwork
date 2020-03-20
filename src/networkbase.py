import networkx as nx 
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
import abc

class NetworkBase:
    """
    Class to represent Arbitrary Bolei-Vincent Network
    """
    __metaclass__ = abc.ABCMeta
    def __init__(self, G, t0, tf):
        self.preprocess(G)
        self.t0 = t0
        self.tf = tf
        # The integrator has not been generated
        self.sim = None
    
    
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
    def f(self,t,y):
        'Right hand side of dydt = f(t,y) '

        non = self.number_of_nodes

        u = y[:non]
        v = y[non:2*non]
        tt = y[2*non:3*non]
        ud = y[3*non:4*non]
        vd = y[4*non:5*non]
        ttd = y[5*non:6*non]

        Fix, Fiy, Mi = self.internal_forces(u,v,tt)
        Fex, Fey, Me = self.external_forces(t)
        Fdx, Fdy, Md = self.damp_forces(ud,vd,ttd) 


        Fx = (Fix+Fex)/self.mass - Fdx
        Fy = (Fiy+Fey)/self.mass - Fdy
        M  = (Mi+Me)/self.J    - Md

        ud,vd,ttd = self.boundary_conditions(t,ud,vd,ttd)

        return np.concatenate((ud, vd, ttd, Fx, Fy, M))

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
    def boundary_conditions(self, t, ud, vd, ttd):
        pass


    def run(self):
        self.sim = integrate.RK45(self.f, t0 = self.t0, y0 = self.y0, t_bound = self.tf, rtol=1e-6, atol=1e-8)
        sim = self.sim
        self.time = []
        self.data = []
        while sim.t < self.tf:
            sim.step()
            self.time.append(sim.t)
            self.data.append(sim.y)
            #percent = sim.t/self.tf
            #print(percent)


    # TODO: Finish the following implementation
    def KE(self,ud,vd,ttd):
        pass

    def PE(self,ud,vd,ttd):
        pass

    def jacobian():
        pass
