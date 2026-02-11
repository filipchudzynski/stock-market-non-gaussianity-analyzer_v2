
import numpy as np
from .base import Detrender
class MovingAverageDetrender(Detrender):
    def detrend_point(self,x,t,s):
        W=self.window_size(s); h=W//2
        return np.mean(x[max(0,t-h):min(len(x),t+h)])
