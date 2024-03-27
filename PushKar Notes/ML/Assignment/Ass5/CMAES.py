import numpy as np
import pdb

def rastrigin(X,n,A=10):
#    second_term = X**2 - A*np.cos(2*np.pi*X)
    
    return n*A + np.sum((X**2 - A * np.cos(2 * np.pi * X)), 1)

def COV(a,mu):
#    mu = np.mean(a,0)
    vcov=  np.dot((a-mu).T,a-mu)/a.shape[0]
    return vcov

def sph(X):
    return np.sum(X**2,axis=1)


if __name__ == "__main__":
    n= 100
    vmin = -5
    vmax = 5
    ## should we have normal or uniform distribution.
    # polulation size
    vPop = 4000
    vratio = .35
    vElit_ratio = int(vPop*vratio)
    mu = np.ones(n)*3
    #vcov = np.ones((n,n))*2
    vcov = np.random.uniform(0,3,(n,n))
    #sigma = np.ones(n)*2
    iter=2000
    for i in range(iter):
        #sample data
        vDat = np.random.multivariate_normal(mu,vcov,vPop)
#        pdb.set_trace()
        vFit = rastrigin(vDat,n)
#        vFit = sph(vDat)
        
        vElit_index = np.argsort(vFit)[:vElit_ratio]
        vPrime = vDat[vElit_index]
        vcov = COV(vPrime,mu)
#        vcov = np.diag(np.diag(vcov))
        mu = vPrime.mean(axis=0)
#        if i==1:
#            pdb.set_trace()
        if i %400==0:
            print(vFit[vElit_index][0])
        


#    vcov = cov(sigma)
    vDat = np.random.multivariate_normal(mu,vcov,vPop)
    vFit = rastrigin(vDat,n)
    top_5 = vFit[np.argsort(vFit)[:5]]
    pdb.set_trace()
