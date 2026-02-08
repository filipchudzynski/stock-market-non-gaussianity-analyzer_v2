
import numpy as np
def generate(N=4096,lambda2=0.05):
    levels=int(np.log2(N)); x=np.ones(N)
    for l in range(levels):
        step=2**l
        w=np.exp(np.random.normal(0,np.sqrt(lambda2),N//step))
        x*=np.repeat(w,step)
    return x/np.std(x)
