import numpy as np
from scipy import stats
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from numpy import linalg as LA
from sklearn import svm
import random
import time
import math

mnist = fetch_mldata('MNIST original')
kf = KFold(n_splits=10, shuffle=True);
X=mnist.data
y=mnist.target
weights= [[0 for i in range(10)] for j in range(15)] #np.zeros((15,10))
#print(weights)
def getR(dists,k):
	sumSquared=0
	for i in dist:
		sumSquared+=i*i
	return math.sqrt(sumSquared/k)

def hiddenFunc(X,i):
	a=LA.norm(X-centers[i])
	b= (a)**2
	c=(b)/r[i]**2
	print(c)
	return math.exp(-(c))


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

clusters=15
kmeans = KMeans(n_clusters=clusters, random_state=0).fit(X)

centers = kmeans.cluster_centers_

# print(centers)
# print(len(centers))

neigh=NearestNeighbors(n_neighbors=20)
neigh.fit(X)

dists,indexs = neigh.kneighbors(X=centers)

r=[]
for dist in dists:
	r.append(getR(dist,20))

print(r[0])
print("look above")


def createWeights(h,o):
	for i in range(0,h):
		for j in range(0,o):
			weights[i][j]=(random.randrange(1,10)*0.01)

def outSum(i,hiddenOuts):
	oSum=0;
	for h in range(0,clusters):
		oSum+=weights[h][i]*hiddenOuts[h]
	return oSum

def nn(inputs):
	count=0;
	for i in inputs:
		hiddenOuts=[]
		outputLayer=[]
		for h in range(0,clusters):
			hiddenOuts.append(hiddenFunc(i,h))
		for o in range(0,10):
			outputLayer.append(outSum(o,hiddenOuts))

		print(outputLayer)
		print (y[count])
		print count
		count+=1


def trainNN(X,y):
	createWeights(clusters,10)
	nn(X)

trainNN(X,y)





