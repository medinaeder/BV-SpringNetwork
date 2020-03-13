import numpy as np
from networkbase import NetworkBase
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as pg
from matplotlib.collections import PatchCollection

from numba import jit, njit


#@njit 
def _forces(u, v, tt, edges, r0):
    """
    args:   u - displacement in x
            v - displacement in y
            tt - theta rotation
            edge definition - connectivity
            r0 - vector for each connectivity
    """
    
    fx = np.zeros_like(u)
    fy = np.zeros_like(v)
    ms  = np.zeros_like(tt)

    # Spring Constants
    k_l = 1.0 
    k_s = 0.02
    k_th = 1e-4/4.

    num_edges = edges.shape[0]
    # Loop through the edges
    for i in range(num_egges):
        e1 = edges[i,0]
        e2 = edges[i,0]
        u1, v1, t1 = u[e1], v[e1], tt[e1]
        u2, v2, t2 = u[e2], v[e2], tt[e2]
            
        # Compute elongation parallel

        # Compute elongation perpendicular




    return fx, fy, ms


class BV_Network(NetworkBase):
    """
    Class to represent the Bolei-Vincent Network
    """
    
    def __init__(self, G, t0, tf):
        NetworkBase.__init__(self, G, t0, tf)
        self.damp_u = 0.01
        self.damp_tt = 0.01 
        self.mass = 1.0
        self.alpha  = 1.8
        self.J =  1./self.alpha**2 * self.mass * 0.5**2 
        self.generate_directions()

    def generate_directions(self):
        G = self.G
        connectivity = []
        vectors =[]
        for node in G.nodes():
            x1,y1 = G.nodes[node]['loc']
            neighbs = [n for n in G.neighbors(node)]
            connectivity.append(neighbs)
            vs = []
            
            # Loop over the edges and find the data for the neighbors
            for i,n in enumerate(G.neighbors(node)):
                x2,y2 = G.nodes[n]['loc']
                vs.append(((x2-x1)*0.5,(y2-y1)*0.5))
            vectors.append(vs)

        self.connectivity = connectivity
        self.vectors = vectors
    
    def internal_forces(self,u,v,tt):
        return _forces(u,v,tt, self.edges, self.e_R)

    def damp_forces(self, ud, vd, ttd):
        du = self.damp_u
        dt = self.damp_tt
        return du*ud, du*vd, dt*ttd 

    def external_forces(self,t):
        non = self.number_of_nodes
        return np.zeros(non), np.zeros(non), np.zeros(non)
        

    def boundary_conditions(self, t, ud, vd, ttd):
        # Apply the velocity
        # loop over nodes 
        delta_y = -0.05
        
        pre_time = 5*n
        load_time = 50*n

        if t > pre_time:
            Veloy = delta_y*n/load_time
            self.damp_u = 0.01
            self.damp_tt = 0.01
        elif t < pre_time:
            Veloy = 0
            self.damp_u = 0.01*2
            self.damp_tt = 0.01*2
        else: 
            Veloy = 0;
        
        G = self.G
        
        # FIXME: Hard coded calc of system size
        for node in G.nodes():
            (x1,y1) = G.nodes[node]['loc']
            if x1==0:
                ud[node] = 0
            if x1==m-1:
                ud[node] = 0
            if y1==0:
                vd[node] = 0
            if y1==n-1:
                vd[node] = Veloy

        return ud, vd, ttd

if __name__ == "__main__":
    import networkx as nx
    m = 8
    n = 8
    
    G = nx.grid_2d_graph(m,n)
    G = nx.convert_node_labels_to_integers(G,label_attribute = "loc")
    #pre_time = 5*n
    #load_time = 50*n
    #total_time = pre_time+load_time
    #problem = BV_Network(G, 0, total_time)
    #import time
    #s = time.time()
    #y = problem.run()
    #e = time.time()
    #np.savetxt("slow.txt", y)
    #print("Total_Time: ", e-s)

    
    
    



    
