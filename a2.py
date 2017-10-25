import numpy as np
from scipy import stats
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import svm
import random


#Basics:
#load mnist dataset
mnist = fetch_mldata('MNIST original')
a = mnist.data.shape
b = mnist.target.shape
c = np.unique(mnist.target)

print(a)
print(b)
print(c)
# Question 1
# Set up:
# spliting the test and training set for k fold cross coreletion with a 10 fold
kf = KFold(n_splits=10);
X=mnist.data
y=mnist.target
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    #print(len(train_index))
    print("TRAIN size:", len(train_index), "TEST size:", len(test_index))
    # in here do the work with testing with the k fold train and test

def backpropogation():
    return 1;


def activation(weights, inputs):
    activation= 0; 
    for i in range(len(weights)-1):
        activation +=weights[i]*inputs[i]
    return activation




class Node(object):
    prevLayer = []
    nextLayer = []
    output = 0

    def output():
        for i in prevLayer:
            output += prevLayer[i].output()



class NeuralNetwork(object):

    def __init__(self,n_nodes,n_layers = 3):
        layers = [n_layers]


    def makeNetwork():
        network = NeuralNetwork()
        return network


print("Question 1: ")



# Question 2
print("Question 2: ")
