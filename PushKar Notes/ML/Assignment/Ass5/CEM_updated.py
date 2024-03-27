from matplotlib import pyplot as plt
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
    
def CEM(vPop,mu,sigma,vratio=.2,iter=400):
    vElit_ratio = int(vPop*vratio)
    vcov = np.diag(sigma)
    vbest =[]
    vworst = []
    n = mu.shape[0]
    for i in range(iter):
        #sample data
        vDat = np.random.multivariate_normal(mu,vcov,vPop)
        #vFit = rastrigin(vDat,n)
        vFit = sph(vDat)
        vindex = np.argsort(vFit)
        vElit_index = vindex[:vElit_ratio]
        vPrime = vDat[vElit_index]
        mu = vPrime.mean(axis=0)
        vcov = COV(vPrime,mu)
        vcov = np.diag(np.diag(vcov))
        vbest.append(vFit[vindex[0]])
        vworst.append(vFit[vindex[-1]])
#        if i %100==0:
 #           print(vFit[vElit_index][0])
    
    return np.array(vbest), np.array(vworst)
    
if __name__ == "__main__":
    
    n= 100
    ## should we have normal or uniform distribution.
    # polulation size
    vPop = 5000
    vratio = .25
#    mu = np.ones(n)*4
    # mean away from the global optima
  #  mu = np.random.uniform(0,1,n)+3
    mu = np.ones(n)*3
    sigma = np.ones(n)*3
    iter=1000
    vbest_avg = 0
    vworst_avg = 0
    for i in range(3):
        vB,vW = CEM(vPop,mu,sigma,vratio,iter)
        vbest_avg +=vB
        vworst_avg+=vW
        print("Best fitness score is {}".format(vB[-1]))
        #print(vB[-1])
    ## vPop=300,vratio= .2,iter= 100,vB=e-6,vW=e-6
    ## vPop
    vbest_avg=vbest_avg/3
    vworst_avg=vworst_avg/3
#    print(len(vbest_avg))
    print("Best fitness average score is {}".format(vbest_avg[-1]))
    print("Worst fitness average score is {}".format(vworst_avg[-1]))
    plt.plot(np.arange(len(vbest_avg)),vbest_avg,label='Best Fitness')
    plt.plot(np.arange(len(vworst_avg)),vworst_avg,label='Worst Fitness')
    plt.xlabel("Number of iterations")
    plt.ylabel("Fitness score")
    plt.legend()
#    plt.show()
    np.save("CEM_best.npy",vbest_avg)
    np.save("CEM_worst.npy",vworst_avg)

    #    vcov = cov(sigma)
#    vDat = np.random.multivariate_normal(mu,vcov,vPop)
#    vFit = rastrigin(vDat,n)
#    top_5 = vFit[np.argsort(vFit)[:5]]
#    pdb.set_trace()


        
