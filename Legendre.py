import numpy as np
import math


def doublefactorial(n):
    res = 1
    while n>=2:
        res *= n
        n -= 2
        
    return res


#def doublefactorial(n):
#    if n <= 1:
#        return 1
#    else:
#        return n*doublefactorial(n-2)
        

#Compute the Associated Legendre polynomials of type 1 with complex z recursively
#l,m > 0, |m| <= l
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
                    

#Compute the Associated Legendre polynomials of type 1 with complex z recursively
#Equivalent to scipy.special.clpmn(m, n, z)*1j, but can take negative m, l
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
    

#Compute the Associated Legendre polynomials of type 1 with complex z non-recursively
#Equivalent to scipy.special.clpmn(m, n, z)    
def LP(l,m,z): 
    p = np.zeros((m+1,l+1), dtype=complex)
    
    p[0,0] = 1
    for l1 in range(m+1):
        p[l1,l1] = ((-1)**l1)*doublefactorial(2*l1-1)*((1-z*z)**(l1/2))
        
    for l1 in range(m+1):
        p[l1,l1+1] = z*(2*l1+1)*p[l1, l1]
    
    for l1 in range(m+1):        
        for l2 in range(m+2,l+1):
            p[l1,l2] = (z*(2*l2-1)*p[l1, l2-1] - (l2+l1-1)*p[l1,l2-2])/(l2-l1)
            
    return p