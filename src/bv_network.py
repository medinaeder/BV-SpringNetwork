import numpy as np
from networkbase import NetworkBase

from numba import njit


@njit 
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

    cos = np.cos
    sin = np.sin

    # Spring Constants
    k_l = 1.0 
    k_s = 0.02
    k_th = 1e-4/4.

    num_edges = edges.shape[0]
    # Loop through the edges
    for i in range(num_edges):
        e1 = edges[i,0]
        e2 = edges[i,1]
        u1, v1, tt1 = u[e1], v[e1], tt[e1]
        u2, v2, tt2 = u[e2], v[e2], tt[e2]

        rx = r0[i,0]
        ry = r0[i,1]


        # Compute the change in length of the spring
        du = u2-u1
        dv = v2-v1
        
        c1 = cos(tt1) 
        s1 = sin(tt1)
        c2 = cos(tt2) 
        s2 = sin(tt2)

        drx = -0.5*((c1+c2-2)*rx-(s1+s2)*ry)
        dry = -0.5*((s1+s2)*rx +(c1+c2-2)*ry)
        
        dl1x = du+drx
        dl1y = dv+dry

        # dl2 = -dl1
        # Compute Force computing- Parallel Perpendicular
        rms = rx*rx+ry*ry # Magnitude of the vector
        Fpar = dl1x*rx+dl1y*ry
        Fper = -ry*dl1x+rx*dl1y

        F1x = (k_l*rx*Fpar-ry*k_s*Fper)/rms
        F1y = (k_l*ry*Fpar+ry*k_s*Fper)/rms

        #Reminder 
        # F2x = -F1x
        # F2y = -F1y
        fx[e1] += F1x
        fy[e1] += F1y

        fx[e2] -= F1x
        fy[e2] -= F1y


        # Torsional Springs
        # M2 = k_th*(tt2-tt1) = M1


        Ms = k_th * (tt1-tt2)
        ms[e1] += Ms
        ms[e2] -= Ms

        # Moment due to force
        # Check forces
        M1 = 0.5*(c1*rx-s1*ry)*F1y-0.5*(s1*rx+c1*ry)*F1x
        M2 = 0.5*(c2*rx-s2*ry)*F1y-0.5*(s2*rx+c2*ry)*F1x

        ms[e1] +=M1
        ms[e2] +=M2


        
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
        m = 20
        n = 20
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
    m = 20
    n = 20
    
    G = nx.grid_2d_graph(m,n)
    pre_time = 5*n
    load_time = 50*n
    total_time = pre_time+load_time
    problem = BV_Network(G, 0, total_time)
    import time
    s = time.time()
    y = problem.run()
    e = time.time()
    print("Final_Time:", problem.sim.t)
    print("Simulation Time ", e-s)
    non = problem.number_of_nodes
    import matplotlib.pyplot as plt
    y = problem.sim.y
    plt.figure(1)
    ax = plt.gca()
    ax.imshow(y[0:non].reshape((m,-1)).T)
    plt.figure(2)
    ax = plt.gca()
    ax.imshow(y[non:2*non].reshape((m,-1)).T)
    plt.figure(3)
    ax = plt.gca()
    ax.imshow(y[2*non:3*non].reshape((m,-1)).T)

    plt.show()
    

    
    
    



    
