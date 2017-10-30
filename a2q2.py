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
kf = KFold(n_splits=5, shuffle=True);
X=mnist.data
y=mnist.target
weights= [[0 for i in range(10)] for j in range(9)] #np.zeros((15,10))
#print(weights)
def getR(dists,k):
	sumSquared=0
	for i in dist:
		sumSquared+=i*i
	return math.sqrt(sumSquared/k)

def hiddenFunc(X,i):
	a=LA.norm(X-centers[i])
	b= (a)**2
	c=(b)/(r[i]**2)
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

clusters=9
print("with clusters n=")
print(clusters)
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



#weights[i][j]= 1/(r[i] *math.sqrt(2*math.pi))
def createWeights(h,o):
	for i in range(0,h):
		for j in range(0,o):
			weights[i][j]= (random.randrange(1,10)*0.01)

def outSum(i,hiddenOuts):
	oSum=0;
	for h in range(0,clusters):
		oSum+=weights[h][i]*hiddenOuts[h]
		# print weights[h][i]
		# print("weight above hidden below")
		# print hiddenOuts[h]
	return oSum
outLayerY=[0,1,2,3,4,5,6,7,8,9]

def nn(inputs,training):
	count=0;
	# for i in inputs:
	hiddenOuts=[]
	outputLayer=[]
	for h in range(0,clusters):
		hiddenOuts.append(hiddenFunc(inputs,h))
	for o in range(0,10):
		outputLayer.append(outSum(o,hiddenOuts))

	count+=1
	if(training):
		for i in range(0,h):
				weights[i]= np.dot(hiddenOuts[i],outLayerY)

	maxOut=-50
	result=0
	for o in range(0,len(outputLayer)):
		if(outputLayer[o]>maxOut):
			maxOut=outputLayer[o]
			result=o

	return result

def trainNN(X,y):
	
	for x in range(0,len(X)):
		out=nn(X[x],True)
		# print("********************")
		# print("********************")
		# print("********************")
		# print("********************")
		# print("result:")
		# print(out)
		# print("should be:")
		# print(y[x])

testingResults=[]
def testNN(X,y):
	correct=0
	whole=0
	for x in range(0,len(X)):
		out=nn(X[x],False)
		print("result:")
		print(out)
		print("should be:")
		print(y[x])
		print(out==y[x])
		if(out==y[x]):
			print("correct")
			correct+=1
		whole+=1
	res=correct/whole
	testingResults.append(res)
	print(res)
	print(correct)
	

createWeights(clusters,10)

for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    #print(len(train_index))
    print("TRAIN size:", len(train_index), "TEST size:", len(test_index))
    # in here do the work with testing with the k fold train and test

    trainNN(X_train,y_train)
    print("Done training")
    testNN(X_test,y_test)

print(testingResults);



