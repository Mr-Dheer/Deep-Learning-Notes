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
vPop = 10000
lr=.01
iter=1000
mu = np.ones(n)*5
vcov = np.random.uniform(0,3,(n,n))
inv_cov  = np.linalg.inv(vcov)



for j in range(iter):
    vDat = np.random.multivariate_normal(mu,vcov,vPop)
#    vFit = rastrigin(vDat,n)
    vFit = sph(vDat)
    grad_mu = 0
    grad_cov=0
    F_mu=0
    F_cov=0
    vdiff = vDat-mu
    for i in range(vPop):
        vD = vdiff[i].reshape((n,-1))
        vgrad_mu = np.matmul(inv_cov,vD)
#        second_term = np.matmul(np.matmul(vD,vD.T),inv_cov)
        second_term = np.matmul(inv_cov,inv_cov)*np.matmul(vD.T,vD)
        vgrad_cov = (-.5*inv_cov) + second_term
        F_mu += np.matmul(vgrad_mu.T,vgrad_mu)
        F_cov += np.matmul(vgrad_cov,vgrad_cov.T)
        grad_mu+= vFit[i]*vgrad_mu
        grad_cov += vFit[i]*vgrad_cov
    F_mu=F_mu/vPop 
    F_cov = F_cov/vPop
    grad_cov = grad_cov/vPop
    grad_mu = grad_mu/vPop
    vcov = vcov-lr*np.matmul(np.linalg.inv(F_cov),grad_cov)
    second_term = lr*grad_mu/F_mu 
    mu = mu.reshape((n,1))-second_term
    inv_cov  = np.linalg.inv(vcov) 
    mu = mu.reshape(-1)
    #   if j%10==0:
    print(vFit[np.argsort(vFit)][:2])
        

        
    
    #inv_cov = np.linalg.inv(vcov)
    #        pdb.set_trace()
    #second_term = (lr*np.matmul(np.linalg.inv(F_mu),grad_mu)).reshape(-1)
    #pdb.set_trace()
#    mu =mu+(lr*np.matmul(np.linalg.inv(F_mu),grad_mu)).reshape(-1)
    #mu =mu+second_term
    #print(mu.shape)
#    mu = mu.reshape(-1)
