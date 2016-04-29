import scipy as sci
from scipy import optimize, special
import numpy as np
import cmath
import math
import time

#alpha=(cos(theta)sin(phi), sin(theta)sin(phi), cos(phi)): unit vectors
def thetaphi(alpha):       
    phi = cmath.acos(alpha[2])
    theta = complex(pi_2)
    sinphi = np.sin(phi)
    
    if(sinphi != 0):
        sintheta = alpha[1]/sinphi
        theta = cmath.asin(sintheta)
        if(alpha[0]/sinphi < 0):            
            theta = np.pi-theta 
         
    return theta, phi
    
    
#Return a vector of spherical harmonic Y   
#l: positive integer
#theta, phi: complex angles
def complexY(l, theta, phi):     
    LP, DLP = sci.special.clpmn(l, l, np.cos(phi), type=2) #Legendre poly
    Yl = np.zeros((2*l+1,), dtype=np.complex)
    
    #Compute Spherical harmonic for wave scattering:
    c = (1j**l)/sqrtpi4
    c1 = 2*l+1
    for m in np.arange(0,l+1):
        Klm = (((-1)**m)*c)*math.sqrt(c1*np.math.factorial(l-m)/np.math.factorial(l+m))
        Yl[m+l] = Klm*np.exp(1j*m*theta)*LP[l,m]
    for m in np.arange(-l,0):
        Yl[m+l] = (-1)**(l-m)*np.conjugate(Yl[-m+l])        
        
    return Yl
    

def complexYMat(theta, phi):
    global n
    YY = np.zeros((n,2*n+1), dtype=np.complex)
    
    for l in range(n):
        Yl = complexY(l, theta, phi)        
        YY[l,:2*l+1] = Yl
        
    return YY
    
    
#Create a cube Y(alpha,l,m) 
#Alpha must contain unit vectors
def complexYCube(Alpha):
    global n
    YYY = np.zeros((Alpha.shape[0],n,2*n+1), dtype=np.complex)
    
    for l in range(Alpha.shape[0]):
        theta, phi = thetaphi(Alpha[l,:])
        YYY[l] = complexYMat(theta, phi)   
            
    return YYY
    
    
#Compute the coeeficients of the scattering solution u   
#alpha: unit vector
#a: the radius of a ball
#kappa: 1-q, q is the potential in Shcrodinger operator: Laplace+1-q
#n: the number of terms for approximation
def ScatteringCoeff(alpha):
    global a, kappa, n
    Al = np.zeros((n,2*n+1), dtype=np.complex) 
    AA = np.zeros((2,2), dtype=np.complex)

    j, jp = special.sph_jn(n-1, kappa*a) #array of Bessel 1st kind and its derivatives
    h, hp = special.sph_yn(n-1, a)       #arrays of Bessel 2nd kind and its derivatives
    
    theta, phi = thetaphi(alpha)    
    
    for l in range(n):
        Y = complexY(l, theta, phi)
        AA[0,0], AA[0,1] = j[l], -h[l]       
        AA[1,0], AA[1,1] = kappa*jp[l], -hp[l]
        for m in np.arange(-l,l+1):
            a0lm = pi4*(1j**l)*np.conjugate(Y[m+l])
            RHS = [a0lm*j[l], a0lm*jp[l]] 
            x = sci.linalg.solve(AA,RHS)   
            #x, info = sci.sparse.linalg.gmres(AA,RHS)   
            Al[l,m+l] = x[1]

    return Al
    

#Compute the scattering amplitude
#alpha, beta: unit vectors
def A(beta, alpha): 
    global n
    
    Al = ScatteringCoeff(alpha)
    theta, phi = thetaphi(beta) 
    
    return np.sum(Al*complexYMat(theta, phi))


def IncidentField(Alpha, X):
    global k
    u0 = np.zeros((Alpha.shape[0],X.shape[0]), dtype=np.complex)
    
    for l in range(Alpha.shape[0]):
        for i in range(X.shape[0]): 
            u0[l,i] = np.exp(1j*np.dot(Alpha[l,:], X[i]))
            
    return u0
    

#Return scattering solution at the point x with incident direction alpha
#x: a point in R^3   
#alpha: incident directions 
def TotalField(x, alpha):
    global n
    r = np.linalg.norm(x)
    h, hp = special.sph_yn(n-1, r) #arrays of Bessel 2nd kind and its derivatives
    theta, phi = thetaphi(x/r) 
    
    YY = complexYMat(theta, phi)  
    U = np.exp(1j*k*np.dot(alpha, x)) + np.sum(h*np.sum(ScatteringCoeff(alpha)*YY,axis=1))
    
    return U 


#Return an array of scattering solution at the point x with different incident 
#direction alphas
#X[i]: a point in R^3   
#Alpha: a vector of different incident directions 
def u(i, Alpha):
    global n, AL
    
    U = np.zeros((Alpha.shape[0],), dtype=np.complex) 
    for l in range(Alpha.shape[0]):
        U[l] = U0[l,i] + np.sum(H[i]*np.sum(AL[l]*YCubeX[i],axis=1))
    
    return U    
    

#Define the scattering function that needs to be minimized    
def fun(Nu):
    global Theta, X, Alpha, delta
                
    ISum = 0
    for i in range(X.shape[0]):        
        S = np.abs(np.exp(-1j*np.dot(Theta, X[i]))*np.sum(Nu*u(i, Alpha))*delta-1)
        ISum +=S*S
    
    return ISum*deltaX
    

#Minimize fun in the annulus a<x<b, x in R^3    
#Theta, Thetap in M={z: z in C, z.z=1}
def Optimize(Theta):
    global n
    Nu = np.zeros((Alpha.shape[0],),dtype=np.complex)
    #nu = np.random.rand(n,2*n+1)/10
    nu=np.ones((n,2*n+1))/3
    
    for l in range(Alpha.shape[0]):
        Nu[l] = np.sum(nu*YCubeA[l])
        
    res = optimize.minimize(fun, Nu, method='BFGS', options={'gtol':1e-3, 'disp': True})  #the best              
    #res = optimize.fmin_cg(fun, nu, gtol=1e-4)    
    #res = optimize.least_squares(fun, nu)
    
    print("Vector Nu =\n", res.x)
    
    return res.x
    
    
#|thetap| -> infinity
#theta, thetap in M={z: z in C, z.z=1}
def ChooseThetaThetapPsi(bigRealNum):
    t = 1.5
    Psi = np.array([0,0,t])
    t2 = cmath.sqrt(1-t*t/4-bigRealNum*bigRealNum)
    Thetap = np.array([bigRealNum,t2,t/2])
    Theta = Thetap - Psi
 
    return Theta, Thetap, Psi
    
    
#Compute the Fourier transform of the recovered potential
#Nu: optimized vector
#|Thetap| -> infinity
#Theta, Thetap in M={z: z in C, z.z=1}
def FourierRecoveredPotential(Nu, Thetap):
    global Alpha, n
    
    #delta = (pi4)/Alpha.shape[0]   #infinitesimal of S^2, unit sphere
    Fq = 0
    for l in range(Alpha.shape[0]):
        Fq += A(Thetap, Alpha[l,:])*Nu[l]
    
    return -pi4*Fq*delta


#Compute the Fourier transform of the potential q analytically
#psi = thetap-theta, |thetap| -> infinity
#theta, thetap in M={z: z in C, z.z=1}
def FourierPotential(q, psi):
    global a
    
    t = np.linalg.norm(psi)
    return ((pi4*q)/(t**3))*(-t*a*np.cos(t*a)+np.sin(t*a))

    
########################## MAIN FUNCTION ###########################  
    
#def main():

ZERO = 10**(-16)
pi_2 = np.pi/2
pi2 = 2*np.pi
pi4 = 4*np.pi
sqrtpi4 = math.sqrt(pi4)

startTime = time.time()     

################ Setting up input parameters ##################
n = 9
print("\nINPUTS:\nThe number of terms that approximate the scattering solution, n =", n)

a = 0.1
print("Radius of a ball in R^3, a =", a)
a1 = a*1.1
#Create an annulus X(a1,b)
b = 1.2
#Volume of the annulus X
VolX = (pi4/3)*(b**3-a1**3)  
#Divide the radius of the annulus from a->b into numRadii parts
numRadii = 2

q = 50
print("The potential in Schrodinger operator (Laplace+k^2-q), q =", q)
k = 1
kappa = k**2 - q

alpha = np.array([0,0,1])
print("Incident field direction, alpha =", alpha)

x = np.array([1,0,0])
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
Alpha = np.vstack((Alpha, np.array([0,0,1])))
Alpha = np.vstack((Alpha, np.array([0,0,-1])))

#infinitesimal of S^2, unit sphere
delta = pi4/Alpha.shape[0]

#Create a grid for the annulus X(a1>a,b)
X = np.zeros((Alpha.shape[0]*numRadii,3),dtype=np.double)
AnnulusRadi = np.linspace(a1, b, numRadii)
i = 0
for R in AnnulusRadi: 
    X[i:i+Alpha.shape[0]] = Alpha*R
    i += Alpha.shape[0]
    
#infinitesimal of X(a1,b), the annulus    
deltaX = VolX/X.shape[0] 

################ Create sample scattering data ##################

U0 = IncidentField(Alpha, X)

#Compute the coefficients of wave scattering solution corresponding to different
#directions of incident wave
AL = np.zeros((Alpha.shape[0],n,2*n+1), dtype=np.complex)
for l in range(Alpha.shape[0]):
    AL[l] = ScatteringCoeff(Alpha[l,:])

AA = A(beta, alpha)
print("\nOUTPUTS:\nScattering amplitude at the point x, A =", AA)

uu = TotalField(x, alpha)
print("Scattering solution at the point x,  u =", uu, "\n")

#arrays of Bessel 2nd kind and its derivatives
H = np.zeros((X.shape[0],n))
HP = H
for i in range(X.shape[0]):
    H[i], HP[i] = special.sph_yn(n-1, np.linalg.norm(X[i])) 
    
YCubeA = complexYCube(Alpha)
XX = X
for i in range(X.shape[0]):
    XX[i] = X[i]/np.linalg.norm(X[i])
YCubeX = complexYCube(XX)

################## Minimize to find vector nu ###################

#theta, thetap in M={z: z in C, z.z=1}
Theta, Thetap, Psi = ChooseThetaThetapPsi(100)

print("Optimizing...")
Nu = Optimize(Theta)
Fq1 = FourierRecoveredPotential(Nu, Thetap)
print("\nFourier(recovered potential):", Fq1)
Fq2 = FourierPotential(q, Psi)
print("Fourier(actual potential q) :", Fq2)
error = np.abs(Fq1-Fq2)/np.abs(Fq2);
print("Relative error:", error)

print("\nTime elapsed:", time.time()-startTime,"seconds")

#if __name__ == "__main__":
#    main()