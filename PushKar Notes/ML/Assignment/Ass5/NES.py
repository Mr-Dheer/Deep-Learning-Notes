from matplotlib import pyplot as plt
import numpy as np
import pdb

#from scipy.stats import multivariate_normal

def rastrigin(X,n,A=10):
#    second_term = X**2 - A*np.cos(2*np.pi*X)
    
    return n*A + np.sum((X**2 - A * np.cos(2 * np.pi * X)), 1)

def COV(a,mu):
#    mu = np.mean(a,0)
    vcov=  np.dot((a-mu).T,a-mu)/a.shape[0]
    return vcov

def sph(X):
    return np.sum(X**2,axis=1)

def NES(vPop,mu,sigma,lr):
    vB = []
    vW = []
    iter=5000
    for i in range(iter):
        s = np.random.normal(size=(vPop,n))
        # gradient of mu computated and stacked as a matrix for the whole dataset instead of doing in a loop
        z = mu + sigma*s
        vFit = rastrigin(z,n)
        #vFit = rastrigin(z,n)
#        print(len(s)==s.shape[0])
        mu -= lr * sigma * 1. / s.shape[0] * np.dot(vFit, s)
        sigma -= lr / 2. * 1. / s.shape[0] * sigma * np.dot(vFit, s ** 2 - 1.)
        vindex = np.argsort(vFit)
        vB.append(vFit[vindex[0]])
        vW.append(vFit[vindex[-1]])
#        if i%100==0:
#            print(vFit[vindex[0]])
    return np.array(vB),np.array(vW)

if __name__ == "__main__":
    n= 100
    # polulation size
    vPop = 8000
    mu = np.ones(n)*3
    sigma = np.ones(n)*4
    lr=.001
    vBest = 0
    vWorst = 0
    for i in range(3):
        print(i)
        vB ,vW = NES(vPop,mu,sigma,lr)
        vBest+=vB
        vWorst+=vW
    vBest=vBest/3
    vWorst=vWorst/3
    plt.plot(np.arange(len(vBest)),vBest,label="Best Fitness")
    plt.plot(np.arange(len(vWorst)),vWorst,label="Worst Fitness")
    plt.xlabel("Number of iterations")
    plt.ylabel("Fitness score")
    plt.legend()
    plt.show()

    np.save("NES_best_rastrigin.npy",vBest)
    np.save("NES_worst_rastrigin.npy",vWorst)


    pdb.set_trace()
