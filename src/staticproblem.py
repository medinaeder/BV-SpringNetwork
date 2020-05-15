import numpy as np
class StaticProblem:
    def __init__(self, problem):
        self.problem = problem
        #self.parameters = parameters
        self.dim = problem.number_of_nodes*3
        self.preprocessBCs()
    
    def preprocessBCs(self):
        # label the indices that we want to apply 0 or the other 
        # Boundary condition at
        top_nodes = []
        bottom_nodes = []
        """
        call static bcs?
        """
        # u,v,tt
        non = self.problem.number_of_nodes
        G = self.problem.G
        for node in G.nodes():
            (x1,y1) = G.nodes[node]['loc']
            if y1==0:
                bottom_nodes.append(node)

            if y1==3:
                top_nodes.append(node)
        
        self.bottom_nodes = bottom_nodes
        self.top_nodes = top_nodes
         

    def residualApplyBCs(self, R, y, r):
        Rup = np.copy(R)
        non = self.problem.number_of_nodes
        
        for node in self.top_nodes:
            Rup[node] = y[node]
            Rup[node+non] = y[node+non]-r 
        for node in self.bottom_nodes:
            Rup[node] = y[node] # Set the node to zero
            Rup[node+non] = y[node+non] # Set the node to zero
             
        return Rup

    def residual(self, y,r):
        """
        y should have the correct boundary conditions imposed on it here
        we will impose the boundary conditions on the residual 
        """
        u,v,tt = self.split(y)
        fiu,fiv,fitt = self.problem.internal_forces(u,v,tt)
        R = np.concatenate((fiu,fiv,fitt))
        R = self.residualApplyBCs(R,y,r)
        return R
    
    def split(self, u):
        non = self.problem.number_of_nodes
        y = np.copy(u)
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

    def jacobian(self,y,r):
        u,v,tt = self.split(y)
        J = self.problem.jacobian(u,v,tt)
        # FIXME: What does numpy do pass value refernce ????
        J = self.jacobianApplyBCs(J)
        return J

    def jacobianApplyBCs(self,J):
        non = self.problem.number_of_nodes
        for node in self.top_nodes:
            J[node,:] = 0 
            J[node,node] = 1
            
            J[node+non,:] = 0
            J[node+non,node+non] = 1
            J[node+2*non,:] = 0
            J[node+2*non,node+2*non] = 1
        
        for node in self.bottom_nodes:
            J[node,:] = 0 
            J[node,node] = 1
            
            J[node+non,:] = 0
            J[node+non,node+non] = 1
            J[node+2*non,:] = 0
            J[node+2*non,node+2*non] = 1
        
        return J
    
    def initial_guess(self):
        non = self.problem.number_of_nodes
        y0 = np.concatenate((np.zeros(non),np.ones(non)*1e-6,np.zeros(non)))
        # Seed random
        # Seed alternating solution
        # Seed uniform compression
        return [np.zeros(3*non)+1e-4]

    
    def stability(self,u,r):
        #e,v = np.linalg.eigh(self.jacobian(u,r))
        return 0.0 #np.min(e)

    def functional(self,y):
        # L2 norm
        return np.dot(y,y)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from bvnetwork import BV_Network
    import networkx as nx
    from deflatedcontinuation import DeflatedContinuation as DefCon
    m = 4
    n = 4
    
    G = nx.grid_2d_graph(m,n)
    model = BV_Network(G)
    problem = StaticProblem(model)
    params = np.linspace(0.0,-0.0001*4,101)
        
    df = DefCon(problem, params[1:], False, 1e-7)
    df.run()
    print("Plotting")
    df.plot_solutions()
    plt.show()
    import IPython; IPython.embed()
    

    # TODO: Test the jacobian 
    # Run more tests
    



