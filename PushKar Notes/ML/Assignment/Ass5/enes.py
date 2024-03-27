import pdb
import numpy as np
from scipy import randn

def rastrigin(X,n,A=10):
#    second_term = X**2 - A*np.cos(2*np.pi*X)
    
    return n*A + np.sum((X**2 - A * np.cos(2 * np.pi * X)), 1)

def sph(X):
    return np.sum(X**2,axis=1)



# initialization
n= 100
A=np.diag(np.ones(n))*2
mu = np.ones(n)*5
vPop = 1000
lr=1e-3
iter=1000
sigma=np.linalg.det(A)**(1/n)
B = A/sigma
lr=.001

for i in range(iter):
    # population generation:
    s = np.random.normal(size =(vPop,n))
#    print(s.shape)
    Z = mu+ sigma*np.dot(s,B)
    vFit = sph(Z)
    vIndex = np.argsort(vFit)
    s = s[vIndex]
    Z = Z[vIndex]
    vFit = vFit[vIndex]
    grad_delta = np.dot(vFit,s)
    grad_M = np.dot(s.T, s*vFit.reshape(vPop,1)) - sum(vFit)*np.eye(n)
    grad_sigma = np.trace(grad_M)/n
    grad_B = grad_M - grad_sigma*np.eye(n)
    
    mu -= lr * sigma * np.dot(B, grad_delta)
    sigma = sigma/np.exp(0.5 * lr* grad_sigma)
    B = np.dot(B, np.exp(0.5 * lr * grad_B))
    if i%200==0:
        print(vFit[:2])
pdb.set_trace()
