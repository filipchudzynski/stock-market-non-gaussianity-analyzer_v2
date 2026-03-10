
import numpy as np

def white_noise(n=100000):
    return np.random.normal(0,1,n)

def brownian_motion(n=100000):
    return np.cumsum(np.random.normal(0,1,n))

def fractional_bm(n=100000, H=0.7):
    noise = np.random.normal(size=n)
    x = np.zeros(n)
    for i in range(1,n):
        x[i] = x[i-1] + noise[i] + H*noise[i-1]
    return x

def simple_mrw(n=100000, lambda2=0.03):
    omega = np.random.normal(0, np.sqrt(lambda2), n)
    sigma = np.exp(omega)
    returns = sigma * np.random.normal(size=n)
    return returns
