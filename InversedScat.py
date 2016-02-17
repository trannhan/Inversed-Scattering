import scipy as sci
from scipy import optimize, special
import numpy as np
import pylab as pl
import math
# For higher precision:
#import mpmath
import time


#Convert a point alpha in S^2 to angles theta and phi
def thetaphi(alpha):      
    phi = math.acos(alpha[2])
    sinphi = math.sqrt(1-alpha[2]**2)
    if(sinphi != 0):
        sintheta = alpha[1]/sinphi
        costheta = alpha[0]/sinphi
        tantheta = sintheta/costheta
        theta = math.atan(tantheta)
    else:
        theta = np.pi/2
    
    return theta, phi
    
    
#Return the sum of spherical harmonic Y   
def Y(l, theta, phi):
    global n
     
    Yl = np.zeros((2*l+1,), dtype=np.complex)
    
    for m in np.arange(-l,l+1):
        Yl[m+l] = sci.special.sph_harm(m,l,theta,phi)
        
    return sum(Yl)


#Return coefficients a0 in the series of e^{i*alpha.x}
def a0(alpha):    
    global n
    
    a_0 = np.zeros((n,), dtype=np.complex)
    #The angles corresponding to the point alpha in S^2
    theta, phi = thetaphi(alpha)
    
    for l in range(n):
        a_0[l] = 4*np.pi*(1j**l)*np.conj(Y(l, theta, phi))
        
    return a_0
    
    
#Compute the coeeficients of the scattering solution u   
#alpha: a point in S^2
#a: the radius of a ball
#kappa: 1-q, q is the potential in Shcrodinger operator: Laplace+1-q
def ScatteringCoeff(alpha, a, kappa):
    global n
    
    Al = np.zeros((n,), dtype=np.complex)
    Bl = np.zeros((n,), dtype=np.complex)    
    AA = np.zeros((2,2), dtype=np.complex)
    
    j, jp = special.sph_jn(n,kappa*a) #array of Bessel 1st kind and its derivatives
    h, hp = special.sph_yn(n,a)      #arrays of Bessel 2nd kind and its derivatives
    a_0 = a0(alpha)
    
    for l in range(n):
        AA[0,0], AA[0,1] = j[l], -h[l]       
        AA[1,0], AA[1,1] = kappa*jp[l], -hp[l]        
        RHS = np.array([a_0[l]*j[l], a_0[l]*jp[l]])  
        x, info = sci.sparse.linalg.gmres(AA,RHS)        
        Al[l], Bl[l] = x[0], x[1]

    return Al, Bl
    
    
#Compute the scattering amplitude
def ScatteringAmplitude(beta, Al):
    global n
    
    A = 0
    theta, phi = thetaphi(beta)
    
    for l in range(n):
        A += Al[l]*Y(l, theta, phi)
    
    return A
    
#Return the scattering solution at the point x with incident direction alpha   
def ScatteringSolution(x, alpha, Al):
    global n

    r = np.linalg.norm(x)
    h, hp = special.sph_yn(n, r) #arrays of Bessel 2nd kind and its derivatives
    x0 = x/r #unit vector in the direction of x
    theta, phi = thetaphi(x0)

    uu = np.exp(1j*np.dot(alpha, x)) #incident field    
    for l in range(n):
        uu += Al[l]*h[l]*Y(l, theta, phi)
        
    return uu

#Return an array of scattering solution at the point x with different incident 
#direction alphas    
def u(x, Alpha):
    global n, a, kappa

    r = np.linalg.norm(x)
    h, hp = special.sph_yn(n, r) #arrays of Bessel 2nd kind and its derivatives
    
    x0 = x/r #unit vector in the direction of x
    theta, phi = thetaphi(x0)
    YY = np.zeros((n,), dtype=np.complex)
    for l in range(n):
        YY[l] = Y(l, theta, phi)

    hYY = h[:len(h)-1]*YY
    
    U = np.zeros((n,), dtype=np.complex)
    for l in range(n):
        U[l] = np.exp(1j*np.dot(Alpha[l,:], x)) + np.sum(AL[l]*hYY)        
    
    return U    
    
#Define the scattering function that needs to be minimized    
def fun(nu):
    global n, theta1, VolX, X, delta, deltaX
    
    ISum = 0
    for x in X:
        ISum += np.abs(np.exp(-1j*np.dot(theta1, x))*np.sum(u(x, Alpha)*nu)*delta-1)**2        
        
    return ISum*deltaX
    
#Minimize fun in the annulus a<x<b, x in R^3    
def FindNu():
    global n
    
    nu = np.zeros((n,))
    res = optimize.minimize(fun, nu, method='BFGS', options={'disp': True})    
    
    return res.x
    
    
########################## MAIN FUNCTION ###########################  
    
#def main():

startTime = time.time()       

a = 1
print("Radius of a ball in R^3, a =", a)
#Create an annulus X(a,b)
b = 2
#Volume of the annulus X
VolX = (4*np.pi/3)*(b**3-a**3)  
#Divide the radius of the annulus from a->b into numRadius parts
numRadius = 3

q = 45
print("The potential in Shcrodinger operator (Laplace+1-q), q =", q)
kappa = 1 - q

n = 20
print("The number of terms that approximates the scattering solution, n =", n)

theta, phi = np.pi/2, np.pi/2
alpha = np.array([np.cos(theta)*np.sin(phi),np.sin(theta)*np.sin(phi),np.cos(phi)])
print("A point in S^2, the surface of the ball, alpha =", alpha)

x = np.array([1,1,1])
print("A point in R^3, x =", x)

beta = x/np.linalg.norm(x)
print("Unit vector in the direction of x, beta =", beta)

#Compute the coefficients of the scattering solution
Al, Bl = ScatteringCoeff(alpha, a, kappa)

A = ScatteringAmplitude(beta, Al)
print("Scattering amplitude at the point x, A =", A)

uu = ScatteringSolution(x, alpha, Al)
print("Scattering solution at the point x, u =", uu)

#Minimize to find nu
rootn = int(np.ceil(np.sqrt(n)))
Theta = np.linspace(0, 2*np.pi, rootn)
Phi = np.linspace(0.1, np.pi-0.1, rootn)
AnnulusR = np.linspace(a*1.1, b*0.9, numRadius)
Alpha = np.zeros((rootn**2,3))
X = np.zeros(((rootn**2)*numRadius,3))

#Create a mesh on the sphere S^2
for l1 in range(rootn):
    for l2 in range(rootn):
        index = l1*rootn + l2
        Alpha[index] = np.array([np.cos(Theta[l1])*np.sin(Phi[l2]),np.sin(Theta[l1])*np.sin(Phi[l2]),np.cos(Phi[l2])])

#Create a grid for the annulus X(a,b)
l1 = 0
for R in AnnulusR: 
    X[l1:l1+Alpha.shape[0]] = Alpha*R
    l1 += Alpha.shape[0]

#Compute the coefficients of wave scattering solution corresponding to different
#directions of incident wave
AL = np.zeros((n,n), dtype=np.complex)
BL = np.zeros((n,n), dtype=np.complex)
for l1 in range(n):
    AL[l1], BL[l1] = ScatteringCoeff(Alpha[l1,:], a, kappa)
        
#Infinitesimals for computing the surface and volume integrals in fun()        
delta = (4*np.pi*a**2)/n
deltaX = VolX/X.shape[0]    
theta1 = beta    
nu = FindNu()

Time = "\nTime elapsed: " + str(time.time()-startTime) + " seconds"
print(Time)
    
#if __name__ == "__main__":
#    main()