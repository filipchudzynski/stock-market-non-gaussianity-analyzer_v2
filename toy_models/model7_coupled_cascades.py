
import numpy as np
def generate(N=4096,lambda2=0.05,coupling=0.3):
    base=np.random.normal(0,np.sqrt(lambda2),N)
    w1=np.exp(base)
    w2=np.exp(coupling*base+np.sqrt(1-coupling**2)*np.random.normal(0,np.sqrt(lambda2),N))
    return np.cumsum(w1*np.random.normal(0,1,N)), np.cumsum(w2*np.random.normal(0,1,N))
