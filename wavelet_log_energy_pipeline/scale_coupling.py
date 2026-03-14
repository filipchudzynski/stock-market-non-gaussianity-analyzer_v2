
import numpy as np
from sklearn.feature_selection import mutual_info_regression

def scale_mutual_information(L):
    '''
    Compute MI between log-energy at different scales
    '''
    n_scales = L.shape[0]
    MI = np.zeros((n_scales,n_scales))

    for i in range(n_scales):
        for j in range(n_scales):
            MI[i,j] = mutual_info_regression(
                L[i].reshape(-1,1), L[j]
            )[0]
    return MI
