
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

def mi_knn(x, y, k=5):
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    xy = np.hstack([x,y])
    tree = cKDTree(xy)
    N = len(x)
    dists, _ = tree.query(xy, k+1)
    eps = dists[:, -1] - 1e-15

    tree_x = cKDTree(x)
    tree_y = cKDTree(y)

    nx = np.array([len(tree_x.query_ball_point(x[i], eps[i])) - 1 for i in range(N)])
    ny = np.array([len(tree_y.query_ball_point(y[i], eps[i])) - 1 for i in range(N)])

    return digamma(k) + digamma(N) - np.mean(digamma(nx+1) + digamma(ny+1))
