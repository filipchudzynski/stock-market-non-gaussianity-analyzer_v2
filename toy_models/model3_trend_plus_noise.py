
import numpy as np
def generate(N=5000,sigma=1.0,trend_type="linear"):
    t=np.arange(N)
    if trend_type=="linear": trend=0.001*t
    elif trend_type=="quadratic": trend=1e-6*t**2
    elif trend_type=="sinusoidal": trend=np.sin(2*np.pi*t/N)
    else: raise ValueError
    return trend+np.random.normal(0,sigma,N)
