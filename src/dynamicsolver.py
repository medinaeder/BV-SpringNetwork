import networkx as nx 
from scipy import integrate
import numpy as np

# Change the interface of this to ode_int

class DynamicSolver:
    def __init__(self, problem, y0,t0,tf):
        self.problem = problem
        self.y0 = y0
        self.t0 = t0
        self.tf = tf

    def f(self,t,y):
        'Right hand side of dydt = f(t,y) '
        problem = self.problem

        non = self.problem.number_of_nodes

        u = y[:non]
        v = y[non:2*non]
        tt = y[2*non:3*non]
        ud = y[3*non:4*non]
        vd = y[4*non:5*non]
        ttd = y[5*non:6*non]

        Fix, Fiy, Mi = problem.internal_forces(u,v,tt)
        Fex, Fey, Me = problem.external_forces(t)
        Fdx, Fdy, Md = problem.damp_forces(ud,vd,ttd) 

        mass = problem.mass
        J = problem.J

        Fx = (Fix+Fex)/mass - Fdx
        Fy = (Fiy+Fey)/mass - Fdy
        M  = (Mi+Me)/J    - Md

        ud,vd,ttd = problem.dynamicBCs(t,ud,vd,ttd)

        return np.concatenate((ud, vd, ttd, Fx, Fy, M))

    def solve(self):
        self.out = integrate.solve_ivp(self.f, t_span = (self.t0,self.tf),  y0 = self.y0, t_eval = np.linspace(self.t0, self.tf, 101))
