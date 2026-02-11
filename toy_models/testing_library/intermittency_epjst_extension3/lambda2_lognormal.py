
import numpy as np
from config import MIN_SAMPLES
def estimate_lambda2_lognormal(inc):
    inc=np.abs(inc); inc=inc[inc>0]
    if len(inc)<MIN_SAMPLES: return np.nan
    return np.var(np.log(inc),ddof=1)
