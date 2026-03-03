
import numpy as np
def brownian_motion(N):
    return np.cumsum(np.random.normal(size=N))
