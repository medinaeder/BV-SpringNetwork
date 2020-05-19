import numpy as np
import scipy as sp
from networkbase import NetworkBase
from numba import njit
import networkx as nx
import matplotlib.pyplot as plt


@njit 
def _forces(u, v, tt, edges, r0,noise):
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
    k_s = 0.02
    k_th = 1e-4/4.

    num_edges = edges.shape[0]
    # Loop through the edges
    for i in range(num_edges):
        k_l = 1.0+noise[i]
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
        # M2 = k_th*(tt2-tt1) = -M1


        M_th = k_th * (tt1-tt2)
        ms[e1] += M_th
        ms[e2] -= M_th

        # Moment due to force
        # Check forces
        M1 = 0.5*(c1*rx-s1*ry)*F1y-0.5*(s1*rx+c1*ry)*F1x
        M2 = 0.5*(c2*rx-s2*ry)*F1y-0.5*(s2*rx+c2*ry)*F1x

        ms[e1] +=M1
        ms[e2] +=M2

    return fx, fy, ms

@njit
def _zerojac(J, edges,non):
    num_edges = edges.shape[0]
    # Loop through the edges
    for i in range(num_edges):
        e1 = edges[i,0]
        e2 = edges[i,1]

        J[e1,e1] = 0
        J[e1,e1+non] = 0
        J[e1,e1+2*non] = 0

        J[e1,e2] = 0
        J[e1,e2+non] = 0
        J[e1,e2+2*non] = 0

        J[e1+non,e1] = 0
        J[e1+non,e1+non] = 0
        J[e1+non,e1+2*non] = 0

        J[e1+non,e2] = 0
        J[e1+non,e2+non] = 0
        J[e1+non,e2+2*non] = 0

        J[e1+2*non,e1] = 0
        J[e1+2*non,e1+non] = 0
        J[e1+2*non,e1+2*non] = 0

        J[e1+2*non,e2] = 0
        J[e1+2*non,e2+non] = 0
        J[e1+2*non,e2+2*non] = 0

        #############################

        J[e2,e1] = 0
        J[e2,e1+non] = 0
        J[e2,e1+2*non] = 0 

        J[e2,e2] = 0
        J[e2,e2+non] = 0
        J[e2,e2+2*non] = 0

        J[e2+non,e1] = 0
        J[e2+non,e1+non] = 0
        J[e2+non,e1+2*non] = 0

        J[e2+non,e2] = 0
        J[e2+non,e2+non] = 0
        J[e2+non,e2+2*non] = 0

        J[e2+2*non,e1] = 0
        J[e2+2*non,e1+non] = 0
        J[e2+2*non,e1+2*non] = 0

        J[e2+2*non,e2] = 0
        J[e2+2*non,e2+non] = 0
        J[e2+2*non,e2+2*non] = 0

@njit
def _jacobian(u, v, tt, edges, r0, J,noise):
    """
    args:   u - displacement in x
            v - displacement in y
            tt - theta rotation
            edge definition - connectivity
            r0 - vector for each connectivity
            J - the Jacobian
    """
    

    cos = np.cos
    sin = np.sin

    # Spring Constants
    k_s = 0.02
    k_th = 1e-4/4.

    num_edges = edges.shape[0]
    non = u.shape[0]
    # Loop through the edges
    for i in range(num_edges):
        k_l = 1.0+noise[i]
        e1 = edges[i,0]
        e2 = edges[i,1]
        u1, v1, tt1 = u[e1], v[e1], tt[e1]
        u2, v2, tt2 = u[e2], v[e2], tt[e2]

        rx = r0[i,0]
        ry = r0[i,1]
        c1 = cos(tt1) 
        s1 = sin(tt1)
        c2 = cos(tt2) 
        s2 = sin(tt2)

        # Set them all to 1 
#        dfx_1du1 = 1
#        dfx_1dv1 = 1
#        dfx_1dtt1 = 1
#
#        dfx_1du2 = 1
#        dfx_1dv2 = 1
#        dfx_1dtt2 = 1
#        
#        dfy_1du1 = 1
#        dfy_1dv1 = 1
#        dfy_1dtt1 = 1
#
#        dfy_1du2 = 1
#        dfy_1dv2 = 1
#        dfy_1dtt2 = 1
#
#        dM1du1 = 1
#        dM1dv1 = 1
#        dM1dtt1 = 1
#
#        dM1du2 = 1
#        dM1dv2 = 1
#        dM1dtt2 = 1
#
#
#
#        dfx_2du1 = 1
#        dfx_2dv1 = 1
#        dfx_2dtt1 = 1
#
#        dfx_2du2 = 1
#        dfx_2dv2 = 1
#        dfx_2dtt2 = 1
#        
#        dfy_2du1 = 1
#        dfy_2dv1 = 1
#        dfy_2dtt1 = 1
#
#        dfy_2du2 = 1
#        dfy_2dv2 = 1
#        dfy_2dtt2 = 1
#
#        dM2du1 = 1
#        dM2dv1 = 1
#        dM2dtt1 = 1
#
#        dM2du2 = 1
#        dM2dv2 = 1
#        dM2dtt2 = 1
        dfx_1du1= (-k_l*rx**2 - k_s*ry**2)/(rx**2 + ry**2)
        dfx_1dv1= (-k_l*rx*ry + k_s*rx*ry)/(rx**2 + ry**2)
        dfx_1dtt1= (k_l*rx*(rx*(0.5*rx*s1 + 0.5*ry*c1) + ry*(-0.5*rx*c1 + 0.5*ry*s1)) - k_s*ry*(rx*(-0.5*rx*c1 + 0.5*ry*s1) - ry*(0.5*rx*s1 + 0.5*ry*c1)))/(rx**2 + ry**2)
        dfx_1du2= (k_l*rx**2 + k_s*ry**2)/(rx**2 + ry**2)
        dfx_1dv2= (k_l*rx*ry - k_s*rx*ry)/(rx**2 + ry**2)
        dfx_1dtt2= (k_l*rx*(rx*(0.5*rx*s2 + 0.5*ry*c2) + ry*(-0.5*rx*c2 + 0.5*ry*s2)) - k_s*ry*(rx*(-0.5*rx*c2 + 0.5*ry*s2) - ry*(0.5*rx*s2 + 0.5*ry*c2)))/(rx**2 + ry**2)
        dfy_1du1= (-k_l*rx*ry + k_s*ry**2)/(rx**2 + ry**2)
        dfy_1dv1= (-k_l*ry**2 - k_s*rx*ry)/(rx**2 + ry**2)
        dfy_1dtt1= (k_l*ry*(rx*(0.5*rx*s1 + 0.5*ry*c1) + ry*(-0.5*rx*c1 + 0.5*ry*s1)) + k_s*ry*(rx*(-0.5*rx*c1 + 0.5*ry*s1) - ry*(0.5*rx*s1 + 0.5*ry*c1)))/(rx**2 + ry**2)
        dfy_1du2= (k_l*rx*ry - k_s*ry**2)/(rx**2 + ry**2)
        dfy_1dv2= (k_l*ry**2 + k_s*rx*ry)/(rx**2 + ry**2)
        dfy_1dtt2= (k_l*ry*(rx*(0.5*rx*s2 + 0.5*ry*c2) + ry*(-0.5*rx*c2 + 0.5*ry*s2)) + k_s*ry*(rx*(-0.5*rx*c2 + 0.5*ry*s2) - ry*(0.5*rx*s2 + 0.5*ry*c2)))/(rx**2 + ry**2)
        dM1du1= -(-k_l*rx**2 - k_s*ry**2)*(0.5*rx*s1 + 0.5*ry*c1)/(rx**2 + ry**2) + (0.5*rx*c1 - 0.5*ry*s1)*(-k_l*rx*ry + k_s*ry**2)/(rx**2 + ry**2)
        dM1dv1= (-k_l*ry**2 - k_s*rx*ry)*(0.5*rx*c1 - 0.5*ry*s1)/(rx**2 + ry**2) - (0.5*rx*s1 + 0.5*ry*c1)*(-k_l*rx*ry + k_s*rx*ry)/(rx**2 + ry**2)
        dM1dtt1= k_th + (-0.5*rx*s1 - 0.5*ry*c1)*(k_l*ry*(rx*(-0.5*rx*(c1 + c2 - 2) + 0.5*ry*(s1 + s2) - u1 + u2) + ry*(-0.5*rx*(s1 + s2) - 0.5*ry*(c1 + c2 - 2) - v1 + v2)) + k_s*ry*(rx*(-0.5*rx*(s1 + s2) - 0.5*ry*(c1 + c2 - 2) - v1 + v2) - ry*(-0.5*rx*(c1 + c2 - 2) + 0.5*ry*(s1 + s2) - u1 + u2)))/(rx**2 + ry**2) - (0.5*rx*s1 + 0.5*ry*c1)*(k_l*rx*(rx*(0.5*rx*s1 + 0.5*ry*c1) + ry*(-0.5*rx*c1 + 0.5*ry*s1)) - k_s*ry*(rx*(-0.5*rx*c1 + 0.5*ry*s1) - ry*(0.5*rx*s1 + 0.5*ry*c1)))/(rx**2 + ry**2) - (0.5*rx*c1 - 0.5*ry*s1)*(k_l*rx*(rx*(-0.5*rx*(c1 + c2 - 2) + 0.5*ry*(s1 + s2) - u1 + u2) + ry*(-0.5*rx*(s1 + s2) - 0.5*ry*(c1 + c2 - 2) - v1 + v2)) - k_s*ry*(rx*(-0.5*rx*(s1 + s2) - 0.5*ry*(c1 + c2 - 2) - v1 + v2) - ry*(-0.5*rx*(c1 + c2 - 2) + 0.5*ry*(s1 + s2) - u1 + u2)))/(rx**2 + ry**2) + (0.5*rx*c1 - 0.5*ry*s1)*(k_l*ry*(rx*(0.5*rx*s1 + 0.5*ry*c1) + ry*(-0.5*rx*c1 + 0.5*ry*s1)) + k_s*ry*(rx*(-0.5*rx*c1 + 0.5*ry*s1) - ry*(0.5*rx*s1 + 0.5*ry*c1)))/(rx**2 + ry**2)
        dM1du2= -(k_l*rx**2 + k_s*ry**2)*(0.5*rx*s1 + 0.5*ry*c1)/(rx**2 + ry**2) + (0.5*rx*c1 - 0.5*ry*s1)*(k_l*rx*ry - k_s*ry**2)/(rx**2 + ry**2)
        dM1dv2= (k_l*ry**2 + k_s*rx*ry)*(0.5*rx*c1 - 0.5*ry*s1)/(rx**2 + ry**2) - (0.5*rx*s1 + 0.5*ry*c1)*(k_l*rx*ry - k_s*rx*ry)/(rx**2 + ry**2)
        dM1dtt2= -k_th - (0.5*rx*s1 + 0.5*ry*c1)*(k_l*rx*(rx*(0.5*rx*s2 + 0.5*ry*c2) + ry*(-0.5*rx*c2 + 0.5*ry*s2)) - k_s*ry*(rx*(-0.5*rx*c2 + 0.5*ry*s2) - ry*(0.5*rx*s2 + 0.5*ry*c2)))/(rx**2 + ry**2) + (0.5*rx*c1 - 0.5*ry*s1)*(k_l*ry*(rx*(0.5*rx*s2 + 0.5*ry*c2) + ry*(-0.5*rx*c2 + 0.5*ry*s2)) + k_s*ry*(rx*(-0.5*rx*c2 + 0.5*ry*s2) - ry*(0.5*rx*s2 + 0.5*ry*c2)))/(rx**2 + ry**2)
        ####################
        dfx_2du1= -(-k_l*rx**2 - k_s*ry**2)/(rx**2 + ry**2)
        dfx_2dv1= -(-k_l*rx*ry + k_s*rx*ry)/(rx**2 + ry**2)
        dfx_2dtt1= -(k_l*rx*(rx*(0.5*rx*s1 + 0.5*ry*c1) + ry*(-0.5*rx*c1 + 0.5*ry*s1)) - k_s*ry*(rx*(-0.5*rx*c1 + 0.5*ry*s1) - ry*(0.5*rx*s1 + 0.5*ry*c1)))/(rx**2 + ry**2)
        dfx_2du2= -(k_l*rx**2 + k_s*ry**2)/(rx**2 + ry**2)
        dfx_2dv2= -(k_l*rx*ry - k_s*rx*ry)/(rx**2 + ry**2)
        dfx_2dtt2= -(k_l*rx*(rx*(0.5*rx*s2 + 0.5*ry*c2) + ry*(-0.5*rx*c2 + 0.5*ry*s2)) - k_s*ry*(rx*(-0.5*rx*c2 + 0.5*ry*s2) - ry*(0.5*rx*s2 + 0.5*ry*c2)))/(rx**2 + ry**2)
        dfy_2du1= -(-k_l*rx*ry + k_s*ry**2)/(rx**2 + ry**2)
        dfy_2dv1= -(-k_l*ry**2 - k_s*rx*ry)/(rx**2 + ry**2)
        dfy_2dtt1= -(k_l*ry*(rx*(0.5*rx*s1 + 0.5*ry*c1) + ry*(-0.5*rx*c1 + 0.5*ry*s1)) + k_s*ry*(rx*(-0.5*rx*c1 + 0.5*ry*s1) - ry*(0.5*rx*s1 + 0.5*ry*c1)))/(rx**2 + ry**2)
        dfy_2du2= -(k_l*rx*ry - k_s*ry**2)/(rx**2 + ry**2)
        dfy_2dv2= -(k_l*ry**2 + k_s*rx*ry)/(rx**2 + ry**2)
        dfy_2dtt2= -(k_l*ry*(rx*(0.5*rx*s2 + 0.5*ry*c2) + ry*(-0.5*rx*c2 + 0.5*ry*s2)) + k_s*ry*(rx*(-0.5*rx*c2 + 0.5*ry*s2) - ry*(0.5*rx*s2 + 0.5*ry*c2)))/(rx**2 + ry**2)
        dM2du1= -(-k_l*rx**2 - k_s*ry**2)*(0.5*rx*s2 + 0.5*ry*c2)/(rx**2 + ry**2) + (0.5*rx*c2 - 0.5*ry*s2)*(-k_l*rx*ry + k_s*ry**2)/(rx**2 + ry**2)
        dM2dv1= (-k_l*ry**2 - k_s*rx*ry)*(0.5*rx*c2 - 0.5*ry*s2)/(rx**2 + ry**2) - (0.5*rx*s2 + 0.5*ry*c2)*(-k_l*rx*ry + k_s*rx*ry)/(rx**2 + ry**2)
        dM2dtt1= -k_th - (0.5*rx*s2 + 0.5*ry*c2)*(k_l*rx*(rx*(0.5*rx*s1 + 0.5*ry*c1) + ry*(-0.5*rx*c1 + 0.5*ry*s1)) - k_s*ry*(rx*(-0.5*rx*c1 + 0.5*ry*s1) - ry*(0.5*rx*s1 + 0.5*ry*c1)))/(rx**2 + ry**2) + (0.5*rx*c2 - 0.5*ry*s2)*(k_l*ry*(rx*(0.5*rx*s1 + 0.5*ry*c1) + ry*(-0.5*rx*c1 + 0.5*ry*s1)) + k_s*ry*(rx*(-0.5*rx*c1 + 0.5*ry*s1) - ry*(0.5*rx*s1 + 0.5*ry*c1)))/(rx**2 + ry**2)
        dM2du2= -(k_l*rx**2 + k_s*ry**2)*(0.5*rx*s2 + 0.5*ry*c2)/(rx**2 + ry**2) + (0.5*rx*c2 - 0.5*ry*s2)*(k_l*rx*ry - k_s*ry**2)/(rx**2 + ry**2)
        dM2dv2= (k_l*ry**2 + k_s*rx*ry)*(0.5*rx*c2 - 0.5*ry*s2)/(rx**2 + ry**2) - (0.5*rx*s2 + 0.5*ry*c2)*(k_l*rx*ry - k_s*rx*ry)/(rx**2 + ry**2)
        dM2dtt2= k_th + (-0.5*rx*s2 - 0.5*ry*c2)*(k_l*ry*(rx*(-0.5*rx*(c1 + c2 - 2) + 0.5*ry*(s1 + s2) - u1 + u2) + ry*(-0.5*rx*(s1 + s2) - 0.5*ry*(c1 + c2 - 2) - v1 + v2)) + k_s*ry*(rx*(-0.5*rx*(s1 + s2) - 0.5*ry*(c1 + c2 - 2) - v1 + v2) - ry*(-0.5*rx*(c1 + c2 - 2) + 0.5*ry*(s1 + s2) - u1 + u2)))/(rx**2 + ry**2) - (0.5*rx*s2 + 0.5*ry*c2)*(k_l*rx*(rx*(0.5*rx*s2 + 0.5*ry*c2) + ry*(-0.5*rx*c2 + 0.5*ry*s2)) - k_s*ry*(rx*(-0.5*rx*c2 + 0.5*ry*s2) - ry*(0.5*rx*s2 + 0.5*ry*c2)))/(rx**2 + ry**2) - (0.5*rx*c2 - 0.5*ry*s2)*(k_l*rx*(rx*(-0.5*rx*(c1 + c2 - 2) + 0.5*ry*(s1 + s2) - u1 + u2) + ry*(-0.5*rx*(s1 + s2) - 0.5*ry*(c1 + c2 - 2) - v1 + v2)) - k_s*ry*(rx*(-0.5*rx*(s1 + s2) - 0.5*ry*(c1 + c2 - 2) - v1 + v2) - ry*(-0.5*rx*(c1 + c2 - 2) + 0.5*ry*(s1 + s2) - u1 + u2)))/(rx**2 + ry**2) + (0.5*rx*c2 - 0.5*ry*s2)*(k_l*ry*(rx*(0.5*rx*s2 + 0.5*ry*c2) + ry*(-0.5*rx*c2 + 0.5*ry*s2)) + k_s*ry*(rx*(-0.5*rx*c2 + 0.5*ry*s2) - ry*(0.5*rx*s2 + 0.5*ry*c2)))/(rx**2 + ry**2)

        J[e1,e1] += dfx_1du1
        J[e1,e1+non] += dfx_1dv1
        J[e1,e1+2*non] += dfx_1dtt1

        J[e1,e2] += dfx_1du2
        J[e1,e2+non] += dfx_1dv2
        J[e1,e2+2*non] += dfx_1dtt2

        J[e1+non,e1] += dfy_1du1
        J[e1+non,e1+non] += dfy_1dv1
        J[e1+non,e1+2*non] += dfy_1dtt1

        J[e1+non,e2] += dfy_1du2
        J[e1+non,e2+non] += dfy_1dv2
        J[e1+non,e2+2*non] += dfy_1dtt2

        J[e1+2*non,e1] += dM1du1
        J[e1+2*non,e1+non] += dM1dv1
        J[e1+2*non,e1+2*non] += dM1dtt1

        J[e1+2*non,e2] += dM1du2
        J[e1+2*non,e2+non] += dM1dv2
        J[e1+2*non,e2+2*non] += dM1dtt2

        #############################

        J[e2,e1] += dfx_2du1
        J[e2,e1+non] += dfx_2dv1
        J[e2,e1+2*non] += dfx_2dtt1

        J[e2,e2] += dfx_2du2
        J[e2,e2+non] += dfx_2dv2
        J[e2,e2+2*non] += dfx_2dtt2

        J[e2+non,e1] += dfy_2du1
        J[e2+non,e1+non] += dfy_2dv1
        J[e2+non,e1+2*non] += dfy_2dtt1

        J[e2+non,e2] += dfy_2du2
        J[e2+non,e2+non] += dfy_2dv2
        J[e2+non,e2+2*non] += dfy_2dtt2

        J[e2+2*non,e1] += dM2du1
        J[e2+2*non,e1+non] += dM2dv1
        J[e2+2*non,e1+2*non] += dM2dtt1

        J[e2+2*non,e2] += dM2du2
        J[e2+2*non,e2+non] += dM2dv2
        J[e2+2*non,e2+2*non] += dM2dtt2

    return 


class BV_Network(NetworkBase):
    """
    Class to represent the Bolei-Vincent Network
    """
    
    def __init__(self, G):
        NetworkBase.__init__(self, G)

        # These have to be characteristics of the netowrk right
        self.damp_u = 0.01
        self.damp_tt = 0.01 
        self.mass = 1.0
        self.alpha  = 1.8
        self.J =  1./self.alpha**2 * self.mass * 0.5**2 
        non = self.number_of_nodes
        self.jac = np.zeros((3*non,3*non))
        np.random.seed(0)
        self.noise = (np.random.rand(self.e_R.shape[0])-0.5)*5e-4
        
        onez = np.ones(non)
        self.MassMatrix = np.diag(np.concatenate((onez,onez,self.J*onez)))
        
        
        # Intialize Solution
        # TODO: Fix to generalize the static and the dynamic
        self.y0 = np.zeros(self.number_of_nodes*6)
        non = self.number_of_nodes
        for i in range(non):
            ipj = i%8+i//8
            self.y0[i]=2e-5*(np.random.rand()-0.5)
            self.y0[non+i]=2e-5*(np.random.rand()-0.5)
            self.y0[2*non+i] = 1e-4*(-1)**(ipj)

    def internal_forces(self,u,v,tt):
        return _forces(u,v,tt, self.edges, self.e_R,self.noise)

    def jacobian(self,u,v,tt):
        _zerojac(self.jac, self.edges,self.number_of_nodes)
        _jacobian(u,v,tt,self.edges,self.e_R,self.jac, self.noise)
        return self.jac
    

    def damp_forces(self, ud, vd, ttd):
        du = self.damp_u
        dt = self.damp_tt
        return du*ud, du*vd, dt*ttd 

    def external_forces(self,t):
        non = self.number_of_nodes
        return np.zeros(non), np.zeros(non), np.zeros(non)

    def dynamicBCs(self, t, ud, vd, ttd):
        m = 8
        n = 8
        delta_y = -0.05
        pre_time = 5*n
        load_time = 50*n
        G = self.G

        if t > pre_time:
            Veloy = delta_y*n/load_time
            self.damp_u = 0.01
            self.damp_tt = 0.01
        elif t < pre_time:
            Veloy = 0
            self.damp_u = 0.01*2
            self.damp_tt = 0.01*2
        
        
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

    def staticBCs(self):
        pass
    
    def plot(self,y):
        plt.figure()
        ax = plt.gca()
        G = self.G

        if len(nx.get_node_attributes(G, "pos")) ==0:
            pos = nx.get_node_attributes(G,"loc")

        else:
            pos = nx.get_node_attributes(G, "pos")
        
        xy = np.array(list(pos.values()))
        assert xy.shape[0] == G.number_of_nodes()
        
        # Loop through the node 
        # if nodes have attribute pos then 
        # else use loc
        # plot the node with the color of the attribute
        # Scatter plot 
        # Add the edge connectivity

        node_collection = ax.scatter(xy[:, 0], xy[:, 1],
                                 s=300,
                                 c=y,
                                 marker="o",
                                 linewidths=None,
                                 edgecolors=None
                                 )
        node_collection.set_zorder(2)
        ax.tick_params(
        axis='both',
        which='both',
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False)
        edgelist = G.edges()
        edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in edgelist])
        from matplotlib.collections import LineCollection
        width = 1
        edge_collection = LineCollection(edge_pos,
                                         colors='k',
                                         linewidths=width,
                                         antialiaseds=(1,),
                                         linestyle='solid',
                                         transOffset=ax.transData,
                                         )

        edge_collection.set_zorder(1)  # edges go behind nodes
        ax.add_collection(edge_collection)
        return


if __name__ == "__main__":
    m = 8
    n = 8
    
    #G = nx.grid_2d_graph(m,n)
    #G = nx.hexagonal_lattice_graph(m,n)
    G = nx.triangular_lattice_graph(m,n)
    pre_time = 5*n
    load_time = 50*n
    total_time = pre_time+load_time
    problem = BV_Network(G)
    #pos = nx.get_node_attributes(problem.G,"loc")
    #y = 2*(np.random.rand(problem.number_of_nodes)-0.5)



    from dynamicsolver import DynamicSolver
    dyn = DynamicSolver(problem,problem.y0, 0, total_time)
    import time
    s = time.time()
    dyn.solve()
    e = time.time()
    print(dyn.out.y.shape)
    print("Simulation Time ", e-s)

    yd = dyn.out.y[:,-1]
    non = problem.number_of_nodes

    problem.plot(yd[:non])
    plt.title("u")
    problem.plot(yd[non:2*non])
    plt.title("v")
    problem.plot(yd[2*non:3*non])
    plt.title(r"\theta")
    plt.show()
    
    #np.savetxt("presentation/time.txt", dyn.out.t) 
    #np.savetxt("presentation/data.txt", dyn.out.y)
    


    
    
    



    
