import scipy as sci
import numpy as np
import sympy as sp
from sympy.mpmath import *
import cmath
import math
import time

#Cannot return complex value due to scipy inable to integrate
def f(r,t,p):
    return mpc.__abs__(cmath.exp(-1j*r*(cos(t)*sin(p)*psi[0] + sin(t)*sin(p)*psi[1] + cos(p)*psi[2]))*r*r*sin(p))
    
def realf(r,t,p):
    return math.cos(r*(cos(t)*sin(p)*psi[0] + sin(t)*sin(p)*psi[1] + cos(p)*psi[2]))*r*r*sin(p)
    
def imagf(r,t,p):
    return -math.sin(r*(cos(t)*sin(p)*psi[0] + sin(t)*sin(p)*psi[1] + cos(p)*psi[2]))*r*r*sin(p)    


#Compute the Fourier transform of the potential q
def FourierPotential(q, a, psi):
    r, t, p = sp.symbols('r, t, p')
    f = sp.exp(-1j*r*(sp.cos(t)*sp.sin(p)*psi[0] + sp.sin(t)*sp.sin(p)*psi[1] + sp.cos(p)*psi[2]))*r*r*sp.sin(p)  
    I = sp.integrate(f, (r, 0, a), (t, 0, 2*sp.pi), (p, 0, sp.pi))
    
    return q*I
  
################# MAIN ####################
mp.dps = 15; mp.pretty = True
#print(mp) 

q = 3  
a = 1
psi = [0,0,10**20]  

####### USING SYMPY ######## VERY FAST
startTime = time.time() 
I1 = FourierPotential(q, a, psi)
print("\nI1 =", I1)
Time = "Time elapsed: " + str(time.time()-startTime) + " seconds"
print(Time)


####### USING SCIPY ######## VERY SLOW, DOES NOT WORK WITH COMPLEX-VALUE FUNCTIONS
#startTime = time.time() 
#I2 = quad(f, [0, a], [0, 2*pi], [0, pi])            #very slow
#print("\nI2 =", I2)
#Time = "Time elapsed: " + str(time.time()-startTime) + " seconds"
#print(Time)


startTime = time.time() 
#I3 = sci.integrate.nquad(realf, [[0, a], [0, 2*np.pi], [0, np.pi]])
#I3 += 1j*sci.integrate.nquad(imagf, [[0, a], [0, 2*np.pi], [0, np.pi]])
I3 = sci.integrate.nquad(f, [[0, a], [0, 2*np.pi], [0, np.pi]])
print("\nI3 =", I3)
Time = "Time elapsed: " + str(time.time()-startTime) + " seconds"
print(Time)


#Reversed order of variables
f1 = lambda p, t, r: mpc.__abs__(cmath.exp(-1j*r*(cos(t)*sin(p)*psi[0] + sin(t)*sin(p)*psi[1] + cos(p)*psi[2]))*r*r*sin(p))
gfun = lambda r: mpf(0)
hfun = lambda r: mpf(2*mp.pi)
qfun = lambda r,t: mpf(0)
rfun = lambda r,t: mpf(mp.pi)

startTime = time.time() 
I4 = sci.integrate.tplquad(f1, 0, a, gfun, hfun, qfun, rfun)    
print("\nI4 =", I4)
Time = "Time elapsed: " + str(time.time()-startTime) + " seconds"
print(Time)
