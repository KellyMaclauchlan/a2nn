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
# for train_index, test_index in kf.split(X):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#     #print(len(train_index))
#     print("TRAIN size:", len(train_index), "TEST size:", len(test_index))
#     # in here do the work with testing with the k fold train and test




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
        self.inputNode = None
        self.correction=None
        self.out=None



    def createWeights(self,prevLayer):
        for i in prevLayer:
            #may need to change initial random values
            self.weights.append([i, (random.randrange(1,10)*0.01)])

class inputNode(object):
    def __init__(self):
        value = None
        self.nextLayer = []

class outputNode(object):
    def __init__(self,target):
        self.weights = []
        self.value = None
        self.output = target
        self.correction=None

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
        print("create input layer")
        i = 0
        while i < n_inputs:
            self.inputLayer.append(inputNode())
            i += 1
        print("connect first layer in input nodes")
        i = 0
        while i < n_nodes:
            self.layers[0][i].inputNode = self.inputLayer[i]
            i += 1

        #give Nodes weights to the next layer
        i = 1
        j = 0
        while i < n_layers:
            j = 0
            while j < n_nodes:
                self.layers[i][j].createWeights(self.layers[i-1])
                j+=1
            i+=1

        print("create output nodes")
        i = 0
        while i <= outputs:
            self.outputLayer.append(outputNode(i))
            i += 1
        print("connect output nodes to last hidden layer ")
        for i in self.outputLayer:
            i.createWeights(self.layers[len(self.layers)-1])


    #dataInstance is one data piece
    #dataTarget is the target output for that data
    def createOutput(self,dataInstance):

        for i in range(0,len(self.inputLayer)):
            self.inputLayer[i].value = dataInstance[i]

        for i in range(0,len(self.layers)):
            for j in range(0,len(self.layers[i])):
                node = self.layers[i][j]
                inputSum = 0
                for k in node.weights:
                    inputSum += k[0].value * k[1]
                node.value = sigmoid(inputSum)

        for i in range(0,len(self.outputLayer)):
            node = self.outputLayer[i]
            for k in node.weights:
                inputSum += k[0].value * k[1]


            self.outputLayer[i].value = sigmoid(inputSum)
        guess = None
        guessCertianty = 0
        for i in self.outputLayer:
            if i.value > guessCertianty:
                guessCertianty = i.value
                guess = i.output

        return(guess)


    def backpropogation(self,correctResult):
        #output layer correction
        for i in self.outputLayer:
            expected= 1 if i.output==correctResult else 0
            out=i.output
            error = expected-out
            i.correction = error*out*(1-out)

        #layer before output, correction 
        for i in range(0,len(self.outputLayer)):
            node = self.outputLayer[i]
            for k in node.weights:
                correctionSum=0;
                for n in k.nextLayer:
                    correctionSum+=k[1]*n.correction
                k[0].correction=node.value*(1-node.value)*correctionSum
                k[1]=k[1]+0.5*node.value*k[0].correction

        #all other layers correction
        for i in range(len(self.layers)-1,1,-1):
            for j in range(0,len(self.layers[i])):
                node = self.layers[i][j]
                inputSum = 0
                for k in node.weights:
                    correctionSum=0;
                if hasattr(k,"nextLayer"):
                    for n in k.nextLayer:
                        correctionSum+=k[1]*n.correction
                    k[0].correction=node.value*(1-node.value)*correctionSum
                    k[1]=k[1]+0.5*node.value*k[0].correction


        
def train(X,y):
    count=0;
    for x in range(len(X)):
        q1Network.createOutput(X[x])
        q1Network.backpropogation(y[x])
        count+=1;
        print("done itteration:")
        print(count)

def test(X,y):
    diffSum=0
    diffCount=0
    for x in range(len(X)):
        diffSum+= y[x]-q1Network.createOutput(X[x]) 
        diffCount+=1
    realDiff=diffSum/diffCount
    print("Done testing result:")
    print(realDiff);


X=mnist.data
y=mnist.target
q1Network = NeuralNetwork(784,9,784)
# print(q1Network.createOutput(X[14600]))
# print(y[14600])
# print(y)
# print(y.size)
print("Question 1: ")
# start_time = time.time()
# #results = run()
# end_time = time.time()
#print ("Overall running time:"), end_time - start_time
for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    #print(len(train_index))
    print("TRAIN size:", len(train_index), "TEST size:", len(test_index))
    # in here do the work with testing with the k fold train and test
    train(X_train,y_train)
    print("Done training")
    test(X_test,y_test)




# Question 2
print("Question 2: ")
