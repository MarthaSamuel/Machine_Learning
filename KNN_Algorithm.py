
# 08 writing KNN algorithm; an alternative to scikit learn version.source @sentdex
# we will calculate euclidean distance using numpy

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings
style.use('fivethirtyeight')

dataset= {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}#2 classes and their features
new_features= [5, 7]

#to visualize the data above
#line 18-20 or 21
'''for i in dataset:
    for ii in datset[i]:
        plt.scatter(ii[0],ii[1], s=100, color=i)'''
'''[[plt.scatter(ii[0],ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0],new_features[1])
plt.show()'''


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than the total voting groups!')
    #we start comparing the new point with other points to find it's class
    distances = []
    for group in data:
        for features in data[group]:
            #euclidean_distance = sqrt((features[0]-predict[0])**2 + (features[1]-predict[1])**2) or for speed
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])
            #print(distances)
    votes = [i[1] for i in sorted(distances) [:k]]
    print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1) [0][0]
    return vote_result

result = k_nearest_neighbors(dataset, new_features, k=3)
print(result)

[[plt.scatter(ii[0],ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0],new_features[1], color = result)
plt.show()