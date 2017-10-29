import numpy as np
from scipy import stats
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import svm
import random
import time



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
