from sympy import *

# Spring-wise pair of dofs
u1,v1,tt1 = symbols('u1 v1 tt1')
u2,v2,tt2 = symbols('u2 v2 tt2')

# Spring Constants
k_l,k_s,k_th = symbols('k_l k_s k_th')

# Spring Vector
rx, ry = symbols('rx ry')


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

# Forces on Node 1 and Node 2
fx_1 = F1x
fy_1 = F1y

fx_2 = -F1x
fy_2 = -F1y

M_th =  k_th * (tt1-tt2)

# Moments on Node 1 and Node 2

M1 = M_th + 0.5*(c1*rx-s1*ry)*F1y-0.5*(s1*rx+c1*ry)*F1x
M2 = -M_th + 0.5*(c2*rx-s2*ry)*F1y-0.5*(s2*rx+c2*ry)*F1x


# Vector of dofs
vs = [u1,v1,tt1,u2,v2,tt2]
# The force vector
FF1 = [fx_1, fy_1, M1]
F1names = ["fx_1", "fy_1", "M1"]
varnames = ["u1","v1","tt1","u2","v2","tt2"]
for i,ff in enumerate(FF1):
    for j,v in enumerate(vs):
        print("d"+F1names[i]+'d'+varnames[j]+"=",diff(ff,v)) 


print("__________________")
FF2 = [fx_2, fy_2, M2]
F2names = ["fx_2", "fy_2", "M2"]
for i,ff in enumerate(FF2):
    for j,v in enumerate(vs):
        print("d"+F2names[i]+'d'+varnames[j]+"=",diff(ff,v)) 







