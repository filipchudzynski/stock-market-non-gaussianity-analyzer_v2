
import numpy as np
from config import MIN_SAMPLES
def estimate_lambda2_lognormal(inc):
    inc=np.abs(inc); inc=inc[inc>0]
    if len(inc)<MIN_SAMPLES: return np.nan
    return np.var(np.log(inc),ddof=1)

def lambda2_lognormal_estimator(inc):
    Z = np.array(inc)
    Y = np.log(Z**2)
    lambda2 = (0.25 * np.var(Y, ddof=1))  # ddof=1 = unbiased
    return lambda2
