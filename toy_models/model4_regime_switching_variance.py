
import numpy as np
def generate(N=5000,sigmas=(0.5,2.0),switch=(2000,3500)):
    x=np.zeros(N); p1,p2=switch; s1,s2=sigmas
    x[:p1]=np.random.normal(0,s1,p1)
    x[p1:p2]=np.random.normal(0,s2,p2-p1)
    x[p2:]=np.random.normal(0,s1,N-p2)
    return x
