
import numpy as np
def generate(N=5000,lambda2=0.05):
    omega=np.random.normal(0,np.sqrt(lambda2),N)
    inc=np.exp(omega)*np.random.normal(0,1,N)
    return np.cumsum(inc)
