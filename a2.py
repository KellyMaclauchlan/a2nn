import numpy as np
from scipy import stats
from sklearn.datasets import fetch_mldata
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
