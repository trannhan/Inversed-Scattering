import scipy as sp
from scipy import optimize, special
import numpy as np
import pylab as pl
import math
# For higher precision:
#import mpmath

#Convert a point alpha in S^2 to angles theta and phi
def thetaphi(alpha):
    global a
    
    alpha = alpha/a    
    
    phi = math.acos(alpha[2])
    sinphi = math.sqrt(1-alpha[2]**2)
    sintheta = alpha[1]/sinphi
    costheta = alpha[0]/sinphi
    tantheta = sintheta/costheta
    theta = math.atan(tantheta)
    
    return theta, phi
    
#Return the sum of spherical harmonic    
def Y(l):
    global n, theta, phi
     
    Yl = np.zeros((2*l+1,), dtype=np.complex)
    
    for m in np.arange(-l,l+1):
        Yl[m+l] = sp.special.sph_harm(m,l,theta,phi)
        
    return sum(Yl)

#Return coefficients a0 in the series of e^{i*alpha.x}
def a0():    
    global alpha, n
    
    a_0 = np.zeros((n,), dtype=np.complex)
    
    for l in range(n):
        a_0[l] = 4*np.pi*(1j**l)*np.conj(Y(l))
        
    return a_0
    
#Compute the coeeficients of the scattering solution u   
def Computebc():
    global n, kappa, a    
    
    b = np.zeros((n,), dtype=np.complex)
    c = np.zeros((n,), dtype=np.complex)    
    A = np.zeros((2,2), dtype=np.complex)
    
    j, jp = special.sph_jn(n,kappa*a) #array of Bessel 1st kind
    h, hp = special.sph_yn(n,a)       #array of Bessel 2nd kind
    a_0 = a0()
    
    for l in range(n):
        A[0,0], A[0,1] = j[l], -h[l]       
        A[1,0], A[1,1] = kappa*jp[l], -hp[l]        
        RHS = np.array([a_0[l]*j[l], a_0[l]*jp[l]])  
        x, info = sp.sparse.linalg.gmres(A,RHS)        
        c[l], b[l] = x[0], x[1]

    return b, c
    
########################## MAIN FUNCTION ###########################    
#def main():
    
#Radius of a ball    
a = 1
#The potential in Helmholtz eq
q_0 = 35
kappa = 1 - q_0
#The number of terms that approximates the scattering solution
n = 50

theta, phi = np.pi/2, np.pi/2
#A point in S^2, the surface of the ball
alpha = a*np.array([np.cos(theta)*np.sin(phi),np.sin(theta)*np.sin(phi),np.cos(phi)])
#The angles corresponding to the point alpha in S^2
theta, phi = thetaphi(alpha)

#Compute the coefficients of the scattering solution
b, c = Computebc()


    
#if __name__ == "__main__":
#    main()