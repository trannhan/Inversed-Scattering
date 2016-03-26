import scipy as sci
from scipy import optimize, special
import numpy as np
#import pylab as pl
#import matplotlib
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import cmath
import math
import time


def thetaphi(alpha):       
    phi = cmath.acos(alpha[2])
    theta = complex(pi_2)
    sinphi = np.sin(phi)

    if(np.abs(sinphi) > ZERO):
        sintheta = alpha[1]/sinphi
        costheta = alpha[0]/sinphi
        if(np.abs(costheta) > ZERO):
            tantheta = sintheta/costheta
            theta = cmath.atan(tantheta)
         
    return theta, phi

    
#Return the sum of spherical harmonic Y   
#l: positive integer
#theta: in [0, 2pi] 
#phi: in [0, pi]
def Y(l, theta, phi):     
    Yl = np.zeros((2*l+1,), dtype=np.complex)
    
    for m in np.arange(-l,l+1):
        Yl[m+l] = sci.special.sph_harm(m,l,theta,phi) #Spherical harmonic
        #Yl += sci.special.sph_harm(m,l,theta,phi)
        
    return sum(Yl)
    
    
#Return the sum of spherical harmonic Y   
#l: positive integer
#theta, phi: complex angles
def complexY(l, theta, phi):     
    LP, DLP = sci.special.clpmn(l, l, np.cos(phi), type=2) #Legendre poly
    Yl = np.zeros((2*l+1,), dtype=np.complex)
    
    #Compute Spherical harmonic for wave scattering:
    for m in np.arange(0,l+1):
        Klm = (((-1)**m)*(1j**l)/math.sqrt(pi4))*math.sqrt((2*l+1)*np.math.factorial(l-m)/np.math.factorial(l+m))
        Yl[m+l] = Klm*np.exp(1j*m*theta)*LP[l,m]
    for m in np.arange(-l,0):
        Yl[m+l] = (-1)**(l+m)*np.conjugate(Yl[-m+l])        
        
    return sum(Yl)    
    
    
#Return a vector of the sum of spherical harmonic Y   
#n: the number of terms for approximation
#alpha: unit vector
def Yvec(n, alpha):
    theta, phi = thetaphi(alpha)
    YY = np.zeros((n,), dtype=np.complex)
    for l in range(n):
        YY[l] = Y(l, theta, phi)    
        
    return YY
    

#Return a vector of the sum of spherical harmonic complexY   
#n: the number of terms for approximation
#alpha: vector in C^3
def complexYvec(n, alpha):
    theta, phi = thetaphi(alpha)
    YY = np.zeros((n,), dtype=np.complex)
    for l in range(n):
        YY[l] = complexY(l, theta, phi)    
        
    return YY
    
    
def complexYMat(n, Alpha):
    YY = np.zeros((n,Alpha.shape[0]), dtype=np.complex)
    
    for k in range(Alpha.shape[0]):
        theta, phi = thetaphi(Alpha[k,:])
        for l in range(n):
            YY[l,k] = complexY(l, theta, phi)    
            
    return YY


#Return coefficients a0 in the series of e^{i*alpha.x}
#alpha: unit vector
#n: the number of terms for approximation
def a0(alpha, n):    
    a_0 = np.zeros((n,), dtype=np.complex)
    #The angles corresponding to the point alpha in S^2
    theta, phi = thetaphi(alpha)
    
    for l in range(n):
        a_0[l] = pi4*(1j**l)*np.conjugate(complexY(l, theta, phi))
        
    return a_0
    
    
#Compute the coeeficients of the scattering solution u   
#alpha: unit vector
#a: the radius of a ball
#kappa: 1-q, q is the potential in Shcrodinger operator: Laplace+1-q
#n: the number of terms for approximation
def ScatteringCoeff(alpha, a, kappa, n):
    Al = np.zeros((n,), dtype=np.complex) 
    AA = np.zeros((2,2), dtype=np.complex)

    j, jp = special.sph_jn(n-1, kappa*a) #array of Bessel 1st kind and its derivatives
    h, hp = special.sph_yn(n-1, a)       #arrays of Bessel 2nd kind and its derivatives
    a_0 = a0(alpha, n)
    
    for l in range(n):
        AA[0,0], AA[0,1] = j[l], -h[l]       
        AA[1,0], AA[1,1] = kappa*jp[l], -hp[l]        
        RHS = [a_0[l]*j[l], a_0[l]*jp[l]] 
        x = sci.linalg.solve(AA,RHS)        
        Al[l] = x[1]

    return Al


#Compute the scattering amplitude
def A(beta, alpha, n): 
    global a, kappa
    
    Al = ScatteringCoeff(alpha, a, kappa, n)
    
    return sum(Al*complexYvec(n, beta))    
    

#Return an array of scattering solution at the point x with different incident 
#direction alphas
#x: a point in R^3   
#Alpha: a vector of different incident directions 
def u(x, Alpha):
    global n, a, AL

    r = np.linalg.norm(x)
    h, hp = special.sph_yn(n-1, r) #arrays of Bessel 2nd kind and its derivatives

    hYY = h*complexYvec(n, x/r)
    
    U = np.zeros((Alpha.shape[0],), dtype=np.complex)
    for l in range(Alpha.shape[0]):
        U[l] = np.exp(1j*np.dot(Alpha[l,:], x)) + sum(AL[l]*hYY)        
    
    return U    
    
    
#Define the scattering function that needs to be minimized    
def fun(nu):
    global n, theta, X, Alpha, YMat
    
    ISum = 0
    for x in X:
        UU = u(x, Alpha)
        coef = np.exp(-1j*np.dot(theta, x))
        Sum = 0
        for l in range(n):
            Sum += nu[l]*sum(UU*YMat[l,:])            
        ISum +=  np.abs(coef*Sum*delta-1)**2
    
    return ISum*deltaX    
    

#Minimize fun in the annulus a<x<b, x in R^3    
#theta, thetap in M={z: z in C, z.z=1}
def Optimize(theta):
    global n
    
    nu = np.random.rand(n,1)
    res = optimize.minimize(fun, nu, method='BFGS', options={'gtol':1e-3, 'disp': True})  #the best              
    #res = optimize.fmin_cg(fun, nu, gtol=1e-4)    
    #res = optimize.least_squares(fun, nu)
    
    return res.x
    

#Return the integral over S^2 of u*Y_l 
#X: the annulus(a1,b)
def p(l,X):
    global Alpha, YMat
    
    P = np.zeros((X.shape[0],), dtype=complex)

    for i in range(X.shape[0]):
        P[i] = sum(u(X[i], Alpha)*YMat[l,:])
        
    return P*delta
 

#Optimize using quadratic form 
def Optimize1(theta):
    global n, X
    B = np.zeros((n,n), dtype=complex)
    P = np.zeros((n,X.shape[0]), dtype=complex)
    RHS = np.zeros((n,), dtype=complex)
    E = np.zeros((X.shape[0],), dtype=complex)    
    deltaX = VolX/X.shape[0]  #infinitesimal of X(a1,b), the annulus
    
    for l in range(n):
        P[l,:] = p(l,X)
        
    for i in range(n):
        for j in range(n):
            B[i,j] = sum(np.conjugate(P[j,:])*P[i,:])
    B *= deltaX
    
    for i in range(X.shape[0]):
        E[i] = np.exp(-1j*np.dot(theta,X[i]))
    
    for i in range(n):
        RHS[i] = sum(E*P[i,:])
    RHS *= deltaX
        
    nu = sci.linalg.solve(np.real(B),np.real(RHS))
    
    return nu
    
    
#Compute the Fourier transform of the recovered potential
#nu: optimized vector
#|thetap| -> infinity
#theta, thetap in M={z: z in C, z.z=1}
def FourierRecoveredPotential(nu, thetap, n):
    global Alpha
    Nu = np.zeros((Alpha.shape[0],), dtype=complex)
    
    delta = (pi4)/n         #infinitesimal of S^2, unit sphere
    Fq = 0
    for l in range(Alpha.shape[0]):
        Nu[l] = sum(nu*YMat[:,l])
        #Nu[l] = sum(nu*complexYvec(n,Alpha[l,:]))
        Fq += A(thetap, Alpha[l,:], n)*Nu[l]
    
    print("Vector Nu =\n", Nu)
    
    return -pi4*Fq*delta
    

#Compute the Fourier transform of the potential q directly
#psi = thetap-theta, |thetap| -> infinity
#theta, thetap in M={z: z in C, z.z=1}
def FourierPotential1(q, a, psi):
    global Alpha, numRadius, n
    
    #Create a grid for the ball B(a)
    Ba = np.zeros((Alpha.shape[0]*numRadius,3), dtype=np.double)
    AnnulusRadi = np.linspace(0, a, numRadius)
    i = 0
    for R in AnnulusRadi: 
        Ba[i:i+Alpha.shape[0]] = Alpha*R
        i += Alpha.shape[0]
    
    deltaBa = (pi4*a**3)/(3*Ba.shape[0])
    ISum = 0    
    for y in Ba:
        ISum += np.exp(-1j*np.dot(psi,y))
        
    return ISum*q*deltaBa    
    
    
#Compute the Fourier transform of the potential q analytically
#psi = thetap-theta, |thetap| -> infinity
#theta, thetap in M={z: z in C, z.z=1}
def FourierPotential(q, a, psi):
    t = np.linalg.norm(psi)
    return ((pi4*q)/(t**3))*(-t*a*np.cos(t*a)+np.sin(t*a))
    
    
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
        

################## Visualize results ###################
#@nb.jit(target='cpu', cache=True)
def Visualize(Matrix):
    R = np.abs(Matrix)
    
    ############## Cartesian plot ##############
    Theta = np.linspace(0, 2*np.pi, n)
    Phi = np.linspace(0, np.pi, n)
    PHI, THETA = np.meshgrid(Phi, Theta)
    
    X1 = R * np.sin(PHI) * np.cos(THETA)
    X2 = R * np.sin(PHI) * np.sin(THETA)
    X3 = R * np.cos(PHI)
    
    N = R/R.max()
    
    #matplotlib.rc('text', usetex=True)
    #matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(12,10))
    im = ax.plot_surface(X1, X2, X3, rstride=1, cstride=1, facecolors=cm.jet(N))
    ax.set_title(r'$|A_l|$', fontsize=20)
    m = cm.ScalarMappable(cmap=cm.jet)
    m.set_array(R)    # Assign the unnormalized data array to the mappable
                      #so that the scale corresponds to the values of R
    fig.colorbar(m, shrink=0.8);
    
    ############## Spherical plot ##############
    # Coordinate arrays for the graphical representation
    x = np.linspace(-np.pi, np.pi, n)
    y = np.linspace(-np.pi/2, np.pi/2, n)
    X, Y = np.meshgrid(x, y)
    
    # Spherical coordinate arrays derived from x, y
    # Necessary conversions to get Mollweide right
    theta = x.copy()    # physical copy
    theta[x < 0] = 2 * np.pi + x[x<0]
    phi = np.pi/2 - y
    PHI, THETA = np.meshgrid(phi, theta)
    
    fig, ax = plt.subplots(subplot_kw=dict(projection='mollweide'), figsize=(10,8))
    im = ax.pcolormesh(X, Y , R)
    #ax.set_xticklabels(xlabels, fontsize=14)
    #ax.set_yticklabels(ylabels, fontsize=14)
    ax.set_title('$|A_l|$', fontsize=20)
    ax.set_xlabel(r'$\theta$', fontsize=20)
    ax.set_ylabel(r'$\phi$', fontsize=20)
    ax.grid()
    fig.colorbar(im, orientation='horizontal');  
      
    
########################## MAIN FUNCTION ###########################  
    
#def main():

ZERO = 10**(-16)
pi_2 = np.pi/2
pi2 = 2*np.pi
pi4 = 4*np.pi

startTime = time.time()     

################ Setting up input parameters ##################
n = 20
print("\nINPUTS:\nThe number of terms that approximate the scattering solution, n =", n)

a = 1
print("Radius of a ball in R^3, a =", a)
a1 = a*1.1
#Create an annulus X(a1,b)
b = 1.2
#Volume of the annulus X
VolX = (pi4/3)*(b**3-a1**3)  
#infinitesimal of S^2, unit sphere
delta = pi4/n 
#Divide the radius of the annulus from a->b into numRadius parts
numRadius = 1

q = 3
print("The potential in Shcrodinger operator (Laplace+1-q), q =", q)
kappa = 1 - q

alpha = [1,0,0]
print("Incident field direction, alpha =", alpha)

x = [1,0,0]
print("A point in R^3, x =", x)

beta = x/np.linalg.norm(x)
print("Direction of x, beta =", beta)

mphi = 3
cphi = np.pi/(mphi+1)
#Create a mesh on the sphere S^2
Alpha = np.zeros((0,3),dtype=np.double)
for i in range(mphi):
    phi = (i+1)*cphi
    mtheta = np.int(mphi + np.abs(phi-pi_2)*mphi)
    for j in range(mtheta):
        theta = j*pi2/mtheta
        Alpha = np.vstack((Alpha, np.array([np.cos(theta)*np.sin(phi),np.sin(theta)*np.sin(phi),np.cos(phi)])))

#Create a grid for the annulus X(a1>a,b)
X = np.zeros((Alpha.shape[0]*numRadius,3),dtype=np.double)
AnnulusRadi = np.linspace(a1, b, numRadius)
i = 0
for R in AnnulusRadi: 
    X[i:i+Alpha.shape[0]] = Alpha*R
    i += Alpha.shape[0]
    
#infinitesimal of X(a1,b), the annulus    
deltaX = VolX/X.shape[0] 

#Compute the coefficients of wave scattering solution corresponding to different
#directions of incident wave
AL = np.zeros((Alpha.shape[0],n), dtype=np.complex)
for l in range(Alpha.shape[0]):
    AL[l] = ScatteringCoeff(Alpha[l,:], a, kappa, n)

################ Create sample scattering data ##################

AA = A(beta, alpha, n)
print("\nOUTPUTS:\nScattering amplitude at the point x, A =", AA)

uu = u(x, Alpha)
print("Scattering solution at the point x, u =\n", uu, "\n")

################## Minimize to find vector nu ###################

#theta, thetap in M={z: z in C, z.z=1}
theta, thetap = ChooseThetaThetap(10)
psi = thetap - theta
YMat = complexYMat(n, Alpha)

nu = Optimize(theta)
Fq1 = FourierRecoveredPotential(nu, thetap, n)
print("\nFourier(recovered potential):", Fq1)
Fq2 = FourierPotential1(q, a, psi)
print("Fourier(actual potential q) :", Fq2)

#Visualize(AL)

print("\nTime elapsed:", time.time()-startTime,"seconds")

#if __name__ == "__main__":
#    main()