# 15 we are making a custom version of the KMeans Algorithm from Scratch, source @sentdex

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11]])

plt.scatter(X[:, 0], X[:, 1], s=150)  # for the 0th element in the X array and 1st element in the X array
plt.show()

colors = 10 * ['g', 'r', 'b', 'c', 'k']


class K_Means:
    # tolerance is how much the centroid is going to move,usually in %change.maxiter is how much we want to run the iteration
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    # with classification we want to know how accurate we are, with clustering we want to make sure those groups exist

    def fit(self, data):

        self.centroids = {}

        for i in range(self.k):  # if k is 2,i is 0,1
            self.centroids[i] = data[i]
        # here we begin the optimization process
        for i in range(self.max_iter):# if we replace the parameter with 1 we see the position of the points in
            # the first pass(the placement of the centroid), with 2 we see how it has moved in
            # the 2nd pass(the adjustment of the centroid) and its % in the console
            self.classifications = {}  # the keys will be the centroids, the values will be
            # the featuresets contained within those values

            for i in range(self.k):
                self.classifications[i] = []  # the key is the centroids, the values are featuresets #
                # contained within those values

            for featureset in data:  # where data is the X
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                pass
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)  # this
                    # #redefines the centroids

                # 15b  KMeans from Scratch
            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tol:
                    print(np.sum((current_centroid - original_centroid) / original_centroid * 100.0)) # this tells us
                    # how many iterations it went through and how large it is.
                    optimized = False  # if any of these movements move more than the tol, we will say we are not
                        # optimized

            if optimized:
                break

    def predict(self, data):
        distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


clf = K_Means()
clf.fit(X)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], marker='o', color='k', s=150, linewidths=5)

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker='x', color=color, s=150, linewidths=5)

#to add some data
unknowns = np.array([[1,3],
                     [8,9],
                     [5,4],
                     [6,4],
                     [0,3],])
for unknown in unknowns:
    classification = clf.predict(unknown)
    plt.scatter(unknown[0], unknown[1], marker='*', color=colors[classification], s=150, linewidth=5)

plt.show()
