
import numpy as np
from scipy.stats import kurtosis
from config import MIN_SAMPLES
def estimate_lambda2(inc):
    if len(inc)<MIN_SAMPLES: return np.nan
    return kurtosis(inc,fisher=True)
