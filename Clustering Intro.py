#13  Clustering Introduction. flat clutering(KMeans) where we tell the machine the number of clusters
# it's a semi-supervised ML method. Usually used for research and finding structure, source @sentdex
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans

X = np.array([[1,2],
              [1.5,1.8],

              [5,8],
              [8,8],
              [1,0.6],
              [9,11]])

#plt.scatter(X[:,0], X[:,1], s= 150)# for the 0th element in the X array and 1st element in the X array
#plt.show()

# def our clf
clf = KMeans(n_clusters =2)#default is 8. this is where we tell the machine the number of clusters we want
clf.fit(X)

centroids = clf.cluster_centers_
labels = clf.labels_#an array of the labels of our features and same index as our features

colors = 10*['g.','r.','b.','c.','k.']
for i in range (len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize =25)#label will be 0 or 1 since we have 2 clusters,this
    #references indexes of colors which is g or r.
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=150, linewidth =5)
plt.show()