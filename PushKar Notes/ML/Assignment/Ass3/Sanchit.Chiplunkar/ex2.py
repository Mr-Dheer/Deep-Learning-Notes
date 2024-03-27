import pdb
import numpy as np


def compute(O,A,B,pi):
    N = len(pi)
    probs = []
    for i in range(N):
        probs.append(pi[i]*B[i][O[0]])
    for t in range(1,len(O)):
        new_probs = []
        for i in range(N):
            p = 0
            for k in range(N):
#                pdb.set_trace()
                p += probs[k]*A[k][i]*B[i][O[t]]
#                pdb.set_trace()
            new_probs.append(p)
        probs = new_probs
    return probs

def jointprob(O,S,A,B,pi):
    N = len(pi)
    prob = pi[S[0]]*B[S[0]][O[0]]
    for i in range(1,len(O)):
        prob = prob*A[S[i-1]][S[i]]*B[S[i]][O[i]]        
    return prob

## 1,2::1,2
pi = [.5,.5]
A  = np.array([[.5,.5],[.75,.25]])
## 1,2:: B : R : Y
B = np.array([[5/11,2/11,4/11],[3/10,4/10,3/10]])
             
c = {"B":0,"R":1,"Y":2}
# encoding for O the observed variable.
#O = ["Y","R","B"]
O = [2,1,0]
# to confirm with hand-written calculatoin
#O = [2,1]
# Encoding for Urns the hidden variable.
S = [0,1,0]


vProb = compute(O,A,B,pi)
vJoint = jointprob(O,S,A,B,pi)

print("Total Probablity is ",sum(vProb))

print("Joint Probablity is ",vJoint)

Ans = vJoint/sum(vProb)
print("Conditional Probablity is",Ans)

