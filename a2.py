import numpy as np 
from scipy import stats 
from sklearn.datasets import fetch_mldata

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
print("Question 1: ")




# Question 2
print("Question 2: ")