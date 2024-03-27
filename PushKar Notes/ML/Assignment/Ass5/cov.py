import numpy as np
import pdb

a = np.random.randn(20,5)
vcov = np.cov(a.T)

def COV(a):
    mu = np.mean(a,0)
    vcov=  np.dot((a-mu).T,(a-mu).conj())/a.shape[0]
    return vcov

vcov2 = COV(a)
print(vcov[0,0])
print(vcov2[0,0])
pdb.set_trace()
