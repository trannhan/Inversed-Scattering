import numpy as np
import math


def doublefactorial(n):
    if n <= 1:
        return 1
    else:
        return n*doublefactorial(n-2)
        
         
#Compute the Associated Legendre polynomials of type 1 with complex z
#|m| <= l
def P(l,m,z):    
    if(np.abs(m)>l):
        return 0
    if(l == 0) and (m==0):
        return 1
    if(l == m):
        return ((-1)**m)*doublefactorial(2*m-1)*((1-z*z)**(m/2))
    if(l-m==1):
        return z*(2*m+1)*P(m, m, z)
    
    return (z*(2*l-1)*P(l-1, m, z) - (l+m-1)*P(l-2, m, z))/(l-m)
        

#Compute the Associated Legendre polynomials of type 1 with complex z
#Equivalent to scipy.special.clpmn(m, n, z, type=2), but can take negative m, l
#|m| <= l
def LegendrePoly(l,m,z):
    if(np.abs(m)>l):
        return 0
    if l<0 and m>=0:
        return P(-l-1, m, z)
    if m<0 and l>=0:
        return ((-1)**m)*(math.factorial(l-m)/math.factorial(l+m))*P(l, -m, z)
    if m<0 and l<0:
        return 0
    
    return P(l,m,z)         