import scipy as sci
from scipy import optimize, special
import numpy as np
import pylab as pl
import cmath
# For higher precision:
#from mpmath import mp
import time


#Convert a point alpha in R^3 to angles theta and phi
def thetaphi(alpha):      
    #alpha /= np.linalg.norm(alpha)  
    #alpha = np.real(alpha)
    
    phi = cmath.acos(alpha[2])
    theta = np.pi/2
    sinphi = cmath.sin(phi)
    
    if(np.abs(sinphi) > ZERO):
        sintheta = alpha[1]/sinphi
        costheta = alpha[0]/sinphi
        if(np.abs(costheta) > ZERO):
            tantheta = sintheta/costheta
            theta = cmath.atan(tantheta)
         
    return theta, phi
    
    
#Return the sum of spherical harmonic Y   
def Y(l, theta, phi):     
    Yl = np.zeros((2*l+1,), dtype=np.complex)
    
    for m in np.arange(-l,l+1):
        Yl[m+l] = sci.special.sph_harm(m,l,theta,phi)
        
    return np.sum(Yl)
    
#Return a vector of the sum of spherical harmonic Y   
#n: the number of terms for approximation
#alpha: unit vector
def Yvec(n, alpha):
    theta, phi = thetaphi(alpha)
    YY = np.zeros((n,), dtype=np.complex)
    for l in range(n):
        YY[l] = Y(l, theta, phi)    
        
    return YY


#Return coefficients a0 in the series of e^{i*alpha.x}
#alpha: unit vector
#n: the number of terms for approximation
def a0(alpha, n):    
    a_0 = np.zeros((n,), dtype=np.complex)
    #The angles corresponding to the point alpha in S^2
    theta, phi = thetaphi(alpha)
    
    for l in range(n):
        a_0[l] = 4*np.pi*(1j**l)*np.conj(Y(l, theta, phi))
        
    return a_0
    
    
#Compute the coeeficients of the scattering solution u   
#alpha: unit vector
#a: the radius of a ball
#kappa: 1-q, q is the potential in Shcrodinger operator: Laplace+1-q
#n: the number of terms for approximation
def ScatteringCoeff(alpha, a, kappa, n):
    Al = np.zeros((n,), dtype=np.complex)
    Bl = np.zeros((n,), dtype=np.complex)    
    AA = np.zeros((2,2), dtype=np.complex)
    
    j, jp = special.sph_jn(n-1, kappa*a) #array of Bessel 1st kind and its derivatives
    h, hp = special.sph_yn(n-1, a)      #arrays of Bessel 2nd kind and its derivatives
    a_0 = a0(alpha, n)
    
    for l in range(n):
        AA[0,0], AA[0,1] = j[l], -h[l]       
        AA[1,0], AA[1,1] = kappa*jp[l], -hp[l]        
        RHS = np.array([a_0[l]*j[l], a_0[l]*jp[l]])  
        x, info = sci.sparse.linalg.gmres(AA,RHS)        
        Al[l], Bl[l] = x[0], x[1]

    return Al, Bl
    
    
#Compute the scattering amplitude
#beta: unit vector
#Al: Scattering coefficients of the scattering amplitude A
#n: the number of terms for approximation
def ScatteringAmplitude(beta, Al, n): 
    return np.sum(Al*Yvec(n, beta))

#Compute the scattering amplitude
def A(beta, alpha, n): 
    global a, kappa
    
    Al, Bl = ScatteringCoeff(alpha, a, kappa, n)
    
    return np.sum(Al*Yvec(n, beta))    
    
#Return the scattering solution at the point x with incident direction alpha   
#x: a point in R^3  
#alpha: unit vector
#Al: Scattering coefficients of the scattering amplitude A
#n: the number of terms for approximation
def ScatteringSolution(x, alpha, Al, n):
    r = np.linalg.norm(x)
    h, hp = special.sph_yn(n-1, r) #arrays of Bessel 2nd kind and its derivatives

    #Return u = incident field + scattered field
    return cmath.exp(1j*np.dot(alpha, x)) + np.sum(Al*h*Yvec(n, x/r)) 


#Return an array of scattering solution at the point x with different incident 
#direction alphas
#x: a point in R^3   
#Alpha: a vector of different incident directions 
def u(x, Alpha):
    global n, a, AL

    r = np.linalg.norm(x)
    h, hp = special.sph_yn(n-1, r) #arrays of Bessel 2nd kind and its derivatives

    hYY = h*Yvec(n, x/r)
    
    U = np.zeros((n,), dtype=np.complex)
    for l in range(n):
        U[l] = cmath.exp(1j*np.dot(Alpha[l,:], x)) + np.sum(AL[l]*hYY)        
    
    return U    
    
#Define the scattering function that needs to be minimized    
def fun(nu):
    global theta, X, delta, deltaX, Alpha
    
    ISum = 0
    for x in X:
        ISum += np.abs(cmath.exp(-1j*np.dot(theta, x))*np.sum(u(x, Alpha)*nu)*delta-1)**2        
        
    return ISum*deltaX
    
#Minimize fun in the annulus a<x<b, x in R^3    
#theta, thetap in M={z: z in C, z.z=1}
def FindOptimizedVec(theta):
    global n
    
    nu = np.zeros((n,), dtype=np.complex)
    res = optimize.minimize(fun, nu, method='BFGS', options={'gtol':1e-16, 'disp': True})  #the best              
    #res = optimize.fmin_cg(fun, nu, gtol=1e-8)    
    #res = optimize.least_squares(fun, nu)
    
    return res
 

#Compute the Fourier transform of the recovered potential
#nu: optimized vector
#|thetap| -> infinity
#theta, thetap in M={z: z in C, z.z=1}
def FourierRecoveredPotential(nu, thetap, n):
    global Alpha
    
    delta = (4*np.pi)/n         #infinitesimal of S^2, unit sphere
    Fq = 0
    for l in range(n):
        Fq += A(thetap, Alpha[l,:], n)*nu[l]
    
    return -4*np.pi*Fq*delta
    

#Compute the Fourier transform of the potential q
#psi = thetap-theta, |thetap| -> infinity
#theta, thetap in M={z: z in C, z.z=1}
def FourierPotential(q, a, psi, n):
    global Alpha, numRadius
    
    #Create a grid for the ball B(a)
    rootn = int(np.ceil(np.sqrt(n)))
    Ba = np.zeros(((rootn**2)*numRadius,3))
    AnnulusRadi = np.linspace(0, a, numRadius)
    l1 = 0
    for R in AnnulusRadi: 
        Ba[l1:l1+Alpha.shape[0]] = Alpha*R
        l1 += Alpha.shape[0]
    
    deltaBa = (4*np.pi*a**3)/(3*Ba.shape[0])
    ISum = 0    
    for y in Ba:
        ISum += cmath.exp(-1j*np.dot(psi,y))
        
    return ISum*q*deltaBa
    
    
#|thetap| -> infinity
#theta, thetap in M={z: z in C, z.z=1}
def ChooseThetaThetap(bigRealNum):
    v = bigRealNum/2
    w = -v
    b1 = 0
    b2 = np.sqrt(w**2 - 1 - b1**2)
    theta = np.array([b1*1j, b2*1j, w], dtype=np.complex)
    thetap = np.array([b1*1j, b2*1j, v], dtype=np.complex)
 
    return theta, thetap
        
    
########################## MAIN FUNCTION ###########################  
    
#def main():

#mp.dps = 16
#print(mp) 
ZERO = 10**(-16)

startTime = time.time()     

################ Setting up input parameters ##################

n = 4
print("\nThe number of terms that approximate the scattering solution, n =", n)

a = 1.0
print("Radius of a ball in R^3, a =", a)

#Create an annulus X(a,b)
b = 1.1
#Volume of the annulus X
VolX = (4*np.pi/3)*(b**3-a**3)  
#Divide the radius of the annulus from a->b into numRadius parts
numRadius = 3

q = 3.0
print("The potential in Shcrodinger operator (Laplace+1-q), q =", q)
kappa = 1 - q

theta, phi = np.pi/2, np.pi/2
alpha = np.array([np.cos(theta)*np.sin(phi),np.sin(theta)*np.sin(phi),np.cos(phi)])
print("Direction of the incident field, alpha =", alpha)

x = np.array([1,1,1], dtype=np.double)
print("A point in R^3, x =", x)

beta = x/np.linalg.norm(x)
print("Unit vector in the direction of x, beta =", beta)

################ Create sample scattering data ##################

#Compute the coefficients of the scattering solution
Al, Bl = ScatteringCoeff(alpha, a, kappa, n)

AA = ScatteringAmplitude(beta, Al, n)
print("\nScattering amplitude at the point x, A =", AA)

uu = ScatteringSolution(x, alpha, Al, n)
print("Scattering solution at the point x, u =", uu, "\n")

################## Minimize to find vector nu ###################

rootn = int(np.ceil(np.sqrt(n)))

#Create a mesh on the sphere S^2
Alpha = np.zeros((rootn**2,3), dtype=np.double)
Theta = np.linspace(0, 2*np.pi, rootn)
Phi = np.linspace(0, np.pi, rootn)
for l1 in range(rootn):
    for l2 in range(rootn):
        index = l1*rootn + l2
        Alpha[index] = np.array([np.cos(Theta[l1])*np.sin(Phi[l2]),np.sin(Theta[l1])*np.sin(Phi[l2]),np.cos(Phi[l2])])

#Create a grid for the annulus X(a,b)
X = np.zeros(((rootn**2)*numRadius,3), dtype=np.double)
AnnulusRadi = np.linspace(a*1.1, b, numRadius)
l1 = 0
for R in AnnulusRadi: 
    X[l1:l1+Alpha.shape[0]] = Alpha*R
    l1 += Alpha.shape[0]

#Compute the coefficients of wave scattering solution corresponding to different
#directions of incident wave
AL = np.zeros((n,n), dtype=np.complex)
BL = np.zeros((n,n), dtype=np.complex)
for l1 in range(n):
    AL[l1], BL[l1] = ScatteringCoeff(Alpha[l1,:], a, kappa, n)
        
#Infinitesimals for computing the surface and volume integrals in fun()        
delta = (4*np.pi)/n         #infinitesimal of S^2, unit sphere
deltaX = VolX/X.shape[0]    #infinitesimal of X(a,b), the annulus

#theta, thetap in M={z: z in C, z.z=1}
#psi = thetap-theta, |thetap| -> infinity
theta, thetap = ChooseThetaThetap(10**17)    
psi = thetap - theta

nu = FindOptimizedVec(theta)

Fq1 = FourierRecoveredPotential(nu.x, thetap, n)
Fq2 = FourierPotential(q, a, psi, n)
print("\nFourier transform of the recovered potential:", Fq1)
print("Fourier transform of the actual potential q: ", Fq2)

Time = "\nTime elapsed: " + str(time.time()-startTime) + " seconds"
print(Time)
    
#if __name__ == "__main__":
#    main()