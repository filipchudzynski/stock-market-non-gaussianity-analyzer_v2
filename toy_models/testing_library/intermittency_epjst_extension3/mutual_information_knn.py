
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.special import digamma
def mutual_information_knn(x,y,k=5):
    data=np.vstack([x,y]).T
    nn=NearestNeighbors(n_neighbors=k+1,metric='chebyshev').fit(data)
    d,_=nn.kneighbors(data)
    eps=d[:,k]
    return digamma(k)
