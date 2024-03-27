import pdb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm 
from mpl_toolkits.mplot3d import Axes3D 



def sph(X):
    return np.sum(X**2,axis=1)

def rastrigin(X,n,A=10):
#    second_term = X**2 - A*np.cos(2*np.pi*X)
    
    return n*A + np.sum((X**2 - A * np.cos(2 * np.pi * X)), 1)


X = np.linspace(-5, 5, 50)     
Y = np.linspace(-5, 5, 50)     
X, Y = np.meshgrid(X, Y) 
vInp = np.vstack((X,Y)).reshape((-1,2))
#Z = rastrigin(vInp,2)
#Z = (X**2 - 10 * np.cos(2 * np.pi * X)) + \
#    (Y**2 - 10 * np.cos(2 * np.pi * Y)) + 20
Z = X**2 + Y**2
#print(Z.shape) 
#pdb.set_trace()
#fig = plt.figure() 
#ax = fig.gca(projection='3d') 
#ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
#  cmap=cm.nipy_spectral, linewidth=0.08,
#  antialiased=True)    
# plt.savefig('rastrigin_graph.png')
plt.contour(X,Y,Z,20,cmap ='RdGy')
plt.show()


vDat=np.random.uniform(-5,5,(100,2))

vrastrigin = rastrigin(vDat,2)
rast_top_5 = vDat[np.argsort(vrastrigin)][:5]
vsphere = sph(vDat)
sphere_top_5 = vDat[np.argsort(vsphere)][:5]
print(np.mean(sphere_top_5,axis = 0))
print(np.mean(rast_top_5,axis = 0))
pdb.set_trace()
