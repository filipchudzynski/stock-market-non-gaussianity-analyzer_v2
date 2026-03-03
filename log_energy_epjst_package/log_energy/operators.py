
import numpy as np

def increment_operator(x, s):
    """Compute increments at scale s."""
    return x[s:] - x[:-s]
