
import numpy as np
from sklearn.metrics import mutual_info_score
def mutual_information(x,y,bins=20):
    x,y=x[~np.isnan(x)],y[~np.isnan(y)]
    return mutual_info_score(x,y)
