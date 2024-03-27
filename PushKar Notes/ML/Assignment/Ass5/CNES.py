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
#        vD = vdiff[i].reshape((n,-1))
        vgrad_mu = np.matmul(inv_cov,vdiff[i])
#        vgrad_mu = np.matmul(inv_cov,vD)

        #vgrad_cov = (inv_cov*-.5 + .5*np.matmul(vdiff[i],vdiff[i].T)*np.matmul(inv_cov,inv_cov))
        vgrad_cov = inv_cov*-.5 + .5*np.matmul(np.matmul(inv_cov,np.matmul(vdiff[i],vdiff[i].T)),inv_cov)
        #        vgrad_cov = inv_cov*-.5 + .5*np.matmul(np.matmul(inv_cov,np.matmul(vD,vD.T)),inv_cov)
        F_mu += np.matmul(vgrad_mu,vgrad_mu.T)
        F_cov += np.matmul(vgrad_cov,vgrad_cov.T)
        grad_mu+= vFit[i]*vgrad_mu
        grad_cov += vFit[i]*vgrad_cov

    F_mu=F_mu/vPop
    F_cov = F_cov/vPop
    grad_cov = grad_cov/vPop
    grad_mu = grad_mu/vPop
    vcov = vcov+lr*np.matmul(np.linalg.inv(F_cov),grad_cov)
    mu =mu+(lr*grad_mu/F_mu)
    inv_cov  = np.linalg.inv(vcov)
    if j%10==0:
        print(vFit[np.argsort(vFit)][:2])
        

pdb.set_trace()
        
    
    #inv_cov = np.linalg.inv(vcov)
    #        pdb.set_trace()
    #second_term = (lr*np.matmul(np.linalg.inv(F_mu),grad_mu)).reshape(-1)
    #pdb.set_trace()
#    mu =mu+(lr*np.matmul(np.linalg.inv(F_mu),grad_mu)).reshape(-1)
    #mu =mu+second_term
    #print(mu.shape)
#    mu = mu.reshape(-1)
