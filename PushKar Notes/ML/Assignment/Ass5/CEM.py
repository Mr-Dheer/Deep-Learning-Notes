import numpy as np
import pdb


def rastrigin(X,n,A=10):
#    second_term = X**2 - A*np.cos(2*np.pi*X)
    
    return n*A + np.sum((X**2 - A * np.cos(2 * np.pi * X)), 1)

def sph(X):
    return np.sum(X**2,axis=1)

def COV(a,mu):
#    mu = np.mean(a,0)
    vcov=  np.dot((a-mu).T,a-mu)/a.shape[0]
    return vcov
    
    
    

if __name__ == "__main__":
    n= 100
    vmin = -5
    vmax = 5
    ## should we have normal or uniform distribution.
    # polulation size
    vPop = 400
    vratio = .20
    vElit_ratio = int(vPop*vratio)
 #   mu = np.random.uniform(vmin,vmax,n)
    mu = np.ones(n)*3
    #sigma = np.random.uniform(vmin,vmax,n)
    sigma = np.ones(n)*4
    vcov = np.diag(sigma)
#    print(vcov)
#    pdb.set_trace()
    iter=1000
    for i in range(iter):
        #sample data
        vDat = np.random.multivariate_normal(mu,vcov,vPop)
        #vFit = rastrigin(vDat,n)
        vFit = sph(vDat)
        vElit_index = np.argsort(vFit)[:vElit_ratio]
        vPrime = vDat[vElit_index]
        mu = vPrime.mean(axis=0)
        vcov = COV(vPrime,mu)
        vcov = np.diag(np.diag(vcov))
        #vcov = np.var(vPrime,0)
        #vcov = np.diag(vcov)

#        vcov = np.diag(np.diag(vcov))
        if i %100==0:
            print(vFit[vElit_index][0])
        


#    vcov = cov(sigma)
    vDat = np.random.multivariate_normal(mu,vcov,vPop)
    vFit = rastrigin(vDat,n)
    top_5 = vFit[np.argsort(vFit)[:5]]
    pdb.set_trace()


        
