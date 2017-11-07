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

mnist.data = [[0	,0, 0],
[0	,1,	1],
[0,	0,	1],
[1,	0,	1],
[0,	1,	0],
[1,	1,	0],
[1,	0,	0],
[1,	1,	1],
[0,	0,	0],
[0,	1,	1],
[0,	0,	1],
[1,	0,	1],
[0,	1,	0],
[1,	1,	0],
[1,	0,	0],
[1,	1,	1],
[0,	0,	0],
[0,	1,	1],
[0,	0,	1],
[1,	0,	1],
[0,	1,	0],
[1,	1,	0],
[1,	0,	0],
[1,	1,	1],
[0,	0,	0],
[0,	1,	1],
[0,	0,	1],
[1,	0,	1],
[0,	1,	0],
[1,	1,	0],
[1,	0,	0],
[1,	1,	1],
[0,	0,	0],
[0,	1,	1],
[0,	0,	1],
[1,	0,	1],
[0,	1,	0],
[1,	1,	0],
[1,	0,	0],
[1,	1,	1],
[0,	0,	0],
[0,	1,	1],
[0,	0,	1],
[1,	0,	1],
[0,	1,	0],
[1,	1,	0],
[1,	0,	0],
[1,	1,	1],
[0,	0,	0],
[0,	1,	1],
[0,	0,	1],
[1,	0,	1],
[0,	1,	0],
[1,	1,	0],
[1,	0,	0],
[1,	1,	1],
[0,	0,	0],
[0,	1,	1],
[0,	0,	1],
[1,	0,	1],
[0,	1,	0],
[1,	1,	0],
[1,	0,	0],
[1,	1,	1],
[0,	0,	0],
[0,	1,	1],
[0,	0,	1],
[1,	0,	1],
[0,	1,	0],
[1,	1,	0],
[1,	0,	0],
[1,	1,	1],
[0,	0,	0]]

mnist.target = [0,
1,
0,
1,
0,
1,
0,
1,
0,
1,
0,
1,
0,
1,
0,
1,
0,
1,
0,
1,
0,
1,
0,
1,
0,
1,
0,
1,
0,
1,
0,
1,
0,
1,
0,
1,
0,
1,
0,
1,
0,
1,
0,
1,
0,
1,
0,
1,
0,
1,
0,
1,
0,
1,
0,
1,
0,
1,
0,
1,
0,
1,
0,
1,
0,
1,
0,
1,
0,
1,
0,
1,
0]
print(a)
print(b)
print(c)
print(d)
# Question 1
# Set up:
# spliting the test and training set for k fold cross coreletion with a 10 fold
kf = KFold(n_splits=10, shuffle=True);
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
        self.layers = []
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
            self.layers[0][i].createWeights(self.inputLayer)
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
        i = 0.0
        while i <= outputs:
            self.outputLayer.append(outputNode(i))
            i += 1.0
        print("connect output nodes to last hidden layer ")
        for i in self.outputLayer:
            i.createWeights(self.layers[len(self.layers)-1])

        self.layers.append(self.outputLayer)


    #dataInstance is one data piece
    #dataTarget is the target output for that data
    def createOutput(self,dataInstance):
        arr = []
        for i in range(0,len(self.inputLayer)):
            self.inputLayer[i].value = dataInstance[i]
        for i in range(0,len(self.layers)):
            for j in range(0,len(self.layers[i])):
                node = self.layers[i][j]
                inputSum = 0
                for k in node.weights:
                    inputSum += k[0].value * k[1]
                node.value = sigmoid(inputSum)
                inputSum = 0
        for i in range(0,len(self.outputLayer)):
            node = self.outputLayer[i]
            for k in node.weights:
                inputSum += k[0].value * k[1]


            self.outputLayer[i].value = sigmoid(inputSum)
            inputSum = 0
        guess = None
        guessCertianty = 0
        count=0;
        for i in self.outputLayer:
            arr.append(i.value)
            if i.value > guessCertianty:
                guessCertianty = i.value
                guess = i.output
        return(guess)


    def backpropogation(self,correctResult):
        #output layer correction
        count=0
        for i in self.outputLayer:
            expected= 1 if count==correctResult else 0
            out=i.value
            error = expected-out
            i.correction = error*out*(1-out)
            count+=1

        # #layer before output, correction
        # for i in range(0,len(self.layers[len(self.layers)-1])-1):
        for t in range(0,len(self.layers[len(self.layers)-1])):
            node = self.layers[len(self.layers)-1][t]
            for j in range(0,len(node.weights)):
                k =node.weights[j]
                correctionSum=0;
                for n in self.outputLayer:
                    correctionSum+=k[1]*n.correction
                # print(k[1])
                # print(k[1]+0.5*node.value*k[0].correction)
                correct=node.value*(1-node.value)*correctionSum
                # print(correct)
                self.layers[len(self.layers)-1][t].weights[j][0].correction=correct
                self.layers[len(self.layers)-1][t].weights[j][1]=k[1]+0.5*node.value*correct
                # print(k[1]+0.5*node.value*k[0].correction)
                # print(k[1])
                # print(self.layers[len(self.layers)-1][t].weights[j][1])


        #all other layers correction
        for i in range(len(self.layers)-2,1,-1):
            for j in range(0,len(self.layers[i])):
                node = self.layers[i][j]
                for f in range(0,len(node.weights)):
                    k =node.weights[f]
                    correctionSum=0;
                    for n in self.layers[i+1]:
                        correctionSum+=k[1]*n.correction
                    correct=node.value*(1-node.value)*correctionSum
                    self.layers[i][j].weights[f][0].correction=correct
                    self.layers[i][j].weights[f][1]=k[1]+0.5*node.value*correct

                # inputSum = 0
                # for k in node.weights:
                #     correctionSum=0;
                # if hasattr(k,"nextLayer"):
                #     for n in k.nextLayer:
                #         correctionSum+=k[1]*n.correction
                #     k[0].correction=node.value*(1-node.value)*correctionSum
                #     k[1]=k[1]+0.5*node.value*k[0].correction



def train(X,y):
    count=0;
    arr = [];
    for x in range(len(X)):
    #for x in range(3000):
        guess = q1Network.createOutput(X[x])
        q1Network.backpropogation(y[x])
        count+=1;
        num0 = 0
        num1 = 0
        num2 = 0
        num3 = 0
        num4 = 0
        num5 = 0
        num6 = 0
        num7 = 0
        num8 = 0
        num9 = 0

        if(guess == 0 ): num0 += 1
        if(guess == 1 ): num1 += 1
        if(guess == 2 ): num2 += 1
        if(guess == 3 ): num3 += 1
        if(guess == 4 ): num4 += 1
        if(guess == 5 ): num5 += 1
        if(guess == 6 ): num6 += 1
        if(guess == 7 ): num7 += 1
        if(guess == 8 ): num8 += 1
        if(guess == 9 ): num9 += 1

#        print(guess)
#        print(y[x])
#        print()
        if (count % 1000 == 0): print(count)
#        print(count)
        # if(guess==y[x]):
        #     print(guess)
        #     print(y[x])
        #     print(q1Network.outputLayer[0].weights[4][1])
    print(num0)
    print(num1)
    print(num2)
    print(num3)
    print(num4)
    print(num5)
    print(num6)
    print(num7)
    print(num8)
    print(num9)


testingResults=[]
def test(X,y):
    diffSum=0
    diffCount=0
    for x in range(len(X)):
    #for x in range(3000):
        result= q1Network.createOutput(X[x])
        if(result==y[x]):
            diffSum+=1
        diffCount+=1
        realDiff=diffSum/diffCount
    print("Done testing result: this percent was correct")
    print(realDiff);
    testingResults.append(realDiff);


X=mnist.data
y=mnist.target
q1Network = NeuralNetwork(3,3,5,1)
# print(q1Network.createOutput(X[14600]))
# print(y[14600])
# print(y)
# print(y.size)
print("Question 1: ")
# start_time = time.time()
# #results = run()
# end_time = time.time()
#print ("Overall running time:"), end_time - start_time
ccc=0

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

print(testingResults);



# Question 2
print("Question 2: ")
