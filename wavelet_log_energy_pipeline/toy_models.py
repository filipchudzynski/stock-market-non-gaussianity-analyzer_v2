
import numpy as np

def white_noise(n=100000):
    return np.random.normal(0,1,n)

def brownian_motion(n=100000):
    return np.cumsum(np.random.normal(0,1,n))

def simple_volatility_cascade(n=100000, lambda2=0.05):
    omega = np.random.normal(0, np.sqrt(lambda2), n)
    sigma = np.exp(omega)
    return sigma*np.random.normal(size=n)
