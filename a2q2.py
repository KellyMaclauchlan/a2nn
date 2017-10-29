import numpy as np
from scipy import stats
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn import svm
import random
import time
import math

mnist = fetch_mldata('MNIST original')
kf = KFold(n_splits=10, shuffle=True);
X=mnist.data
y=mnist.target

def getR(dists,k):
	sumSquared=0
	for i in dist:
		sumSquared+=i*i
	return math.sqrt(sumSquared/k)

def hiddenFunc(X,i):
	return math.exp(-(((X-centers[i])**2)/r[i]**2))


class outputNode(object):
    #randomly disperse node onto plane
    def __init__(self,xRange, yRange):
        x = random.randrange(0, xRange)
        y = random.randrange(0, yRange)

class rbfNetwork(object):
    def __init__(self,n_outputs,xRange,yRange):
        self.hiddenLayer = []
        for i in range(0,n_outputs):
            self.hiddenLayer.append(outputNode(xRange,yRange))


net = rbfNetwork(10,784,256)

kmeans = KMeans(n_clusters=15, random_state=0).fit(X)

centers = kmeans.cluster_centers_

print(centers)
print(len(centers))

neigh=NearestNeighbors(n_neighbors=20)
neigh.fit(X)

dists,indexs = neigh.kneighbors(X=centers)

r=[]
for dist in dists:
	r.append(getR(dist,20))

print(r)





