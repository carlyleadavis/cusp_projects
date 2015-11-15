###
#
# Code for Visualizations for ADS Foundations Project
# Ideological Effects of Gerrymandering
#
# Carlyle Davis
# 11/15/2015
###

import pandas as pd
from sklearn import decomposition
from sklearn.cluster import KMeans
import contextlib
import math
import numpy as np
import matplotlib.pyplot as pl
#from __future__ import print_function
from mpl_toolkits.mplot3d import Axes3D
#import enthought.mayavi.mlab as mylab

#http://stackoverflow.com/questions/2891790/pretty-printing-of-numpy-array
#set a function to manage the print options of specific arrays
@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield
    np.set_printoptions(**original)


df = pd.read_csv('/Users/carlyle/Documents/ADS/Foundations/data/gerry_features_norm.csv')

data= df.as_matrix(columns = None)
#delte the index and the Geo id2 column
data = data[:,1:-1]
print(data.shape)

pca = decomposition.PCA(n_components = 3)
pca.fit(data)
X = pca.transform(data)

#printout of the variance explained by each column
with printoptions(precision=3, suppress=True):
    print('PCA VARIANCE WITH FEATURES NORMALIZED:\n', pca.explained_variance_ratio_)

#get the weights of each of the features by vector space
i = np.identity(data.shape[1])
#unformatted version of the PCA weights
coef = pca.transform(i)

#formatted version of the PCA weights
weights = pd.DataFrame(coef, columns=['PC-1', 'PC-2', 'PC-3'],
                       index=['max_unemp', 'max_race', 'max_jobs', 'max_ed', 'ks_inc','CookScore', 'spaceindx'])

print('PCA PARAMETERS:\n', pca.get_params(),'\n')

print('PCA WEIGHTS: \n', weights)

###
#
# PCA of Data
#
###
fig1 = pl.figure(figsize = (12,12)) # Make a plotting figure
ax = Axes3D(fig1) # use the plotting figure to create a Axis3D object.
pltData = [X[:, 0], X[:, 1], X[:, 2]]
ax.scatter(pltData[0], pltData[1], pltData[2], 'bo') # make a scatter plot of blue dots from the data

# make simple, bare axis lines through space:
xAxisLine = ((min(pltData[0]), max(pltData[0])), (0, 0), (0,0)) # 2 points make the x-axis line at the data extrema along x-axis
ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r') # make a red line for the x-axis.

yAxisLine = ((0, 0), (min(pltData[1]), max(pltData[1])), (0,0)) # 2 points make the y-axis line at the data extrema along y-axis
ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r') # make a red line for the y-axis.

zAxisLine = ((0, 0), (0,0), (min(pltData[2]), max(pltData[2]))) # 2 points make the z-axis line at the data extrema along z-axis
ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r') # make a red line for the z-axis.

# label the axes
ax.set_xlabel("PC-1 (x-axis)")
ax.set_ylabel("PC-2 (y-axis)")
ax.set_zlabel("PC-3 (z-axis)")
ax.set_title("PCA Analysis of Gerrymandering components")
pl.show()


###
#
# Clustering of Data
#
###
pl.clf()
fig2 = pl.figure(figsize = (12,12)) # Make a plotting figure
ax = Axes3D(fig2)
pl.cla()

#number of clusters
def rule_of_thumb(x):
    #number of clusters by the rule of thumb, as taken from wikipedia.
    #rounds up after finds cluster
    clusters = int(math.ceil(np.sqrt(x/2)))
    return clusters

#clusters = rule_of_thumb(len(X))
clusters = 14

Kmeans_est = KMeans(n_clusters=clusters, n_init = 1000)

Kmeans_est.fit(X)
Kmeans_labels = Kmeans_est.labels_

ax.scatter(pltData[0], pltData[1], pltData[2], c=Kmeans_labels.astype(np.float))

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])


# make simple, bare axis lines through space:
xAxisLine = ((min(pltData[0]), max(pltData[0])), (0, 0), (0,0)) # 2 points make the x-axis line at the data extrema along x-axis
ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r') # make a red line for the x-axis.

yAxisLine = ((0, 0), (min(pltData[1]), max(pltData[1])), (0,0)) # 2 points make the y-axis line at the data extrema along y-axis
ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r') # make a red line for the y-axis.

zAxisLine = ((0, 0), (0,0), (min(pltData[2]), max(pltData[2]))) # 2 points make the z-axis line at the data extrema along z-axis
ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r') # make a red line for the z-axis.


ax.set_xlabel('PC-1')
ax.set_ylabel('PC-2')
ax.set_zlabel('PC-3')
ax.set_title("Clustering of Gerrymandering components, n = {0}".format(clusters))

pl.show()