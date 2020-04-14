import numpy as np

class StaticSolver:
    def __init__(self, problem):
        self.problem = problem
        #self.parameters = parameters

    def applybcs(self, y):
        """
        call static bcs?
        """
        # u,v,tt
        non = self.problem.number_of_nodes
        G = problem.G
        for node in G.nodes():
            (x1,y1) = G.nodes[node]['loc']
            # TODO Fix Hard Coded Calcs
            if y1==0:
                y[node] = 0
                y[node+non] = 0

            if y1==7:
                y[node] = 0
                y[node+non] = 0.05*8
        return y

    def residual(self,y):
        """
        y should have the correct boundary conditions imposed on it here
        we will impose the boundary conditions on the residual 
        """
        u,v,tt = self.split(y)
        fiu,fiv,fitt = self.problem.internal_forces(u,v,tt)

        return np.concatenate((fiu, fiv, fitt))
    
    def split(self, y):
        non = self.problem.number_of_nodes
        return y[:non], y[non:2*non], y[2*non:3*non]

    def picard(self):
        pass

    def fd_jacobian(self,y):
        """
        Computes the finite difference jacobian to compare to our analytical jac
        """
        res0 = self.residual(y)
        eps = 1e-6
        dofs = y.shape[0]
        jac_approx = np.zeros((dofs,dofs))
        for i in range(dofs):
            y_temp = np.copy(y)
            y_temp[i]+=eps

            r2 = self.residual(y_temp)
            dr = (r2-res0)/eps
            for j in range(dofs):
                jac_approx[j,i] = dr[j]
        
        return jac_approx

    def newton(self):
        pass


if __name__ == "__main__":
    from bvnetwork import BV_Network
    import networkx as nx
    m = 8
    n = 8
    
    G = nx.grid_2d_graph(m,n)
    problem = BV_Network(G)
    non = problem.number_of_nodes
    x,y,z = np.zeros(non),np.zeros(non),np.zeros(non)
    problem.jacobian(x,y,z)
    stat = StaticSolver(problem)
    Jac_approx = stat.fd_jacobian(np.zeros(3*non))
    #w,v = np.linalg.eig(stat.fd_jacobian(np.zeros(3*non)))
    w,v = np.linalg.eig(stat.problem.jac)
    print(np.linalg.norm(Jac_approx-stat.problem.jac))
    # Add a test to make sure that the jacobian is properly approximated
    # Also want to check that all the imaginary values are at machien precision

    #print(w)
    #print(w.shape)
    #print(v.shape)
    import matplotlib.pyplot as plt
    v0 = np.imag(v[:,-1])
    print(v0)
    non = 64
    problem.plot(v0[:non])
    problem.plot(v0[non:2*non])
    problem.plot(v0[2*non:3*non])
    #plt.imshow(problem.jac)
    plt.show()


    #StaticSolver(problem)



