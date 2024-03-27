import pdb
import numpy as np

def viterbi(A,B,O,pi):
    #prob at initial state
    N = len(pi)
    probs = []
    vOrder1 = []
    vOrder2 = []
    vOrder = [[0],[1]]
    for i in range(N):
        # first color is yellow and so therefore we take the 1st index for the Emission matrix.
        probs.append(pi[i]*B[i][O[0]])
    #print(probs)
    for t in range(1,len(O)):
        new_probs = []
        for d in range(N):
            prob_temp=[]
            for s in range(N):
                p = probs[s]*A[s][d]*B[d][O[t]]
                prob_temp.append(p)
            vmax = np.amax(prob_temp)
            vind = np.where(prob_temp == vmax)[0][0]
            # changed
            if d==0:
                vOrder1.append(vind)
            else:
                vOrder2.append(vind)
            new_probs.append(vmax)
        probs = new_probs
#        pdb.set_trace()
    vProb = np.amax(probs)
    vInd = np.where(probs==vProb)[0][0]
    #pdb.set_trace()
    if vInd==0:
        vOrder1.append(vInd)
        return vProb,vOrder1
    else:
        vOrder2.append(vInd)
        return vProb,vOrder2
    
#    return probs,vOrder1,vOrder2


pi = [.5,.5]
A  = np.array([[.5,.5],[.75,.25]])
## 1,2:: B : R : Y
B = np.array([[5/11,2/11,4/11],[3/10,4/10,3/10]])
## .01239
# encoding for O the observed variable.
#O = ["R","Y","B"]
O = [1,2,0]
#O = [2,1]
vprob, vl1 = viterbi(A,B,O,pi)
print(vprob)
print(vl1)
             
