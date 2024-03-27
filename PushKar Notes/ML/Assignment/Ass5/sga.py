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
vPop = 1000
lr=.01
iter=1000
mu = np.ones(n)*5
vcov = np.random.uniform(0,3,(n,n))
inv_cov  = np.linalg.inv(vcov)



for j in range(iter):
#    vFit = rastrigin(vDat,n)
    grad_mu = 0
    grad_cov=0
    for i in range(vPop):
#        vD = vdiff[i].reshape((n,-1))
#        vgrad_mu = np.matmul(inv_cov,vdiff[i])
        vDat = np.random.multivariate_normal(mu,vcov,1)
        vdiff = vDat-mu
        vFit = sph(vDat)
        vgrad_mu = np.matmul(vdiff,inv_cov)

        vgrad_cov = (inv_cov*-.5 + .5*np.matmul(vdiff,vdiff.T)*np.matmul(inv_cov,inv_cov))
        #vgrad_cov = inv_cov*-.5 + .5*np.matmul(np.matmul(inv_cov,np.matmul(vdiff[i],vdiff[i].T)),inv_cov)
        #        vgrad_cov = inv_cov*-.5 + .5*np.matmul(np.matmul(inv_cov,np.matmul(vD,vD.T)),inv_cov)
        grad_mu+= vFit*vgrad_mu
        grad_cov += vFit*vgrad_cov
    
#    pdb.set_trace()
    grad_cov = grad_cov/vPop
    grad_mu = (grad_mu/vPop).reshape(-1)
    vcov = vcov-lr*grad_cov
    mu =mu-(lr*grad_mu)
    inv_cov  = np.linalg.inv(vcov)
    #    if j%10==0:
    print(mu[:5])
        

        
    
    #inv_cov = np.linalg.inv(vcov)
    #        pdb.set_trace()
    #second_term = (lr*np.matmul(np.linalg.inv(F_mu),grad_mu)).reshape(-1)
    #pdb.set_trace()
#    mu =mu+(lr*np.matmul(np.linalg.inv(F_mu),grad_mu)).reshape(-1)
    #mu =mu+second_term
    #print(mu.shape)
#    mu = mu.reshape(-1)
