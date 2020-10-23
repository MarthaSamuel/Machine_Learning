
# 02   How to program best fit slope (m),source @sentdex

import numpy as np
from statistics import mean
import matplotlib.pyplot as plt

'''
this is a python list
xs = [1,2,3,4,5,6]
ys = [5,4,6,5,6,7]

plt.scatter(xs,ys)
plt.show()'''

'''here we change our data to a numpy array'''
xs = np.array([1,2,3,4,5,6], dtype=np.float64)
ys = np.array([5,4,6,5,6,7], dtype=np.float64)

def best_fit_slope(xs,ys):
    m = ((( mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)**2)- mean(xs**2)))

    return m

m = best_fit_slope(xs,ys)
print(m)


