import numpy as np
from scipy import stats
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import svm
import random
import time


#Basics:
#load mnist dataset
mnist = fetch_mldata('MNIST original')
a = mnist.data.shape
b = mnist.target.shape
c = np.unique(mnist.target)
d = mnist.DESCR
print(a)
print(b)
print(c)
print(d)
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

def sigmoid(x):
    return 1/(1+np.exp(-x))

def deSigmoid(y):
    return y*(1-y)

def output(node):
    return sigmoid(activation(node.weights,node.inputs))

def correct(out,error):
    return error*out*(1-out)

def error(out,expected):
    return expected-out

def backCorrection(weights,lastCorrect,out):
    correctionSum=0
    for i in range(len(weights)-1):
        correctionSum +=weights[i]*lastCorrect
    return out*(1-out) * correctionSum

def newWeight(weight,out,correct):
    return weight+.5*out*correct




class Node(object):
    def __init__(self):
        #contains pairs of nodes and weights
        self.weights = []
        #contains nodes to pass inputs to
        self.nextLayer = []
        self.value = None



    def createWeights(self,prevLayer):
        for i in prevLayer:
            #may need to change initial random values
            self.weights.append([i, (random.randrange(1,10)*0.01)])





class NeuralNetwork(object):
    #n_inputs = the number of inputs should equal the n_nodes
    #n_layers is hidden layers
    def __init__(self,n_inputs, outputs, n_nodes,n_layers = 3):
        #create nodes in each layer
        self.layers = [[]]
        self.inputLayer = []
        self.outputLayer = []

        i = 0
        while i < n_layers:
            j = 0
            self.layers.append([])
            while j < n_nodes:
                self.layers[i].append(Node())
                j += 1
            i += 1
        #give Nodes weights to the next layer
        i = 0
        j = 0
        while i < n_layers:
            j = 0
            while j < n_nodes:
                self.layers[i][j].createWeights(self.layers[i])
                j+=1
            i+=1

    #dataInstance is one data piece
    #dataTarget is the target output for that data
    def test(dataInstance,dataTarget):

        for i in dataInstance:
            j = 0
            self.inputLayer[j] = i
            j+=1







q1Network = NeuralNetwork(784,9,784)
print("Question 1: ")
start_time = time.time()
results = run()
end_time = time.time()
print ("Overall running time:"), end_time - start_time
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    #print(len(train_index))
    print("TRAIN size:", len(train_index), "TEST size:", len(test_index))
    # in here do the work with testing with the k fold train and test



# Question 2
print("Question 2: ")
