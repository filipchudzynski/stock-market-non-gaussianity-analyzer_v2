import numpy as np

def generate(N=5000,sigma=1.0): return np.cumsum(np.random.normal(0,sigma,N))
