from numpy import *
from scipy import special
import pylab as py
from mpl_toolkits.mplot3d import axes3d as p3

phi = linspace(0, 2*pi, 25)
theta = linspace(-pi/2, pi/2, 50)
ax = []
ay = []
az = []
R = 2.5

for t in theta:
    polar = float(t)
    for k in phi:
        azim = float(k)
        sph = special.sph_harm(0,5,azim,polar) # Y(m,l,phi,theta)
        modulation = 0.2 * abs(sph)
        r = R * ( 1 + modulation)
        x = r*cos(polar)*cos(azim)
        y = r*cos(polar)*sin(azim)
        z = r*sin(polar)
        ax.append(x)
        ay.append(y)
        az.append(z)
        
fig=py.figure()
f = p3.Axes3D(fig)
f.set_xlabel('X')
f.set_ylabel('Y')
f.set_zlabel('Z')
f.scatter3D(ax,ay,az)
py.show()
