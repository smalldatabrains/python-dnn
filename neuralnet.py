#dependencies---------------------------------------------------------------------------------------

import numpy as np #matrix multiplication + maths
import pandas as pd#data management
import os

#network definitions--------------------------------------------------------------------------------
# Z : output vector for each layer
# X : input vector for each layer
# W : Weights matrix
# a : values vector after applying the sigmoid function to Z, a becomes the new input vector for the next layer
# y : real value vector of the ouput
# yth : theoritical value vector calculated by our network
# e : error vector between y and yth
# J : costfunction, return single value

def normalize(X):
	return X/np.amax(X)

def sigmoid(z):
	return 1.0/(1.0+np.exp(-z))

def softmax(z):
	softmax = np.exp(z-np.max(z))/np.sum(np.exp(z-np.max(z)),axis=1,keepdims=True)
	return softmax
		
def dsigmoid(z):
	return np.exp(-z)/(1+np.exp(-z))**2

def Weights(k,l):
	return np.random.randn(k,l)

def bias(l):
	return np.random.rand(l) #it is better to start with positive values for bias

def create_network(input,n_hidden,n_neurons,output):
	W=dict()
	b=dict()
	for i in range(n_hidden+1):
		if i==0:
			W[i]=Weights(input.shape[1],n_neurons[i])
			b[i]=bias(n_neurons[i])
		elif i==n_hidden:
			W[i]=Weights(n_neurons[i-1],2)
			b[i]=bias(2)
		else:
			W[i]=Weights(n_neurons[i-1],n_neurons[i])
			b[i]=bias(n_neurons[i])
	return W,b

def forward(input,W,b):
	for layer in range(len(W)):
		if layer==0:
			Yth=np.dot(input,W[layer])+b[layer]
			Yth=sigmoid(Yth)
		elif layer==len(W)-1:
			Yth=np.dot(Yth,W[layer])+b[layer]
			Yth=softmax(Yth)
		else:
			Yth=np.dot(Yth,W[layer])+b[layer]
			Yth=sigmoid(Yth)
	return Yth

def cost(Yth,Yreal):
	J=np.sum((Yth-Yreal)**2)/(2*len(Yreal))
	return J

def gradient(Yth,Yreal,X,J,W):
	gradient=dict()
	delta=dict()
	Z=dict()
	a=dict()
	for layer in range(len(W)):
		if layer==0:
			Z[layer]=np.dot(X,W[layer])
			a[layer]=sigmoid(Z[layer])
		else:
			Z[layer]=np.dot(a[layer-1],W[layer])
			a[layer]=sigmoid(Z[layer])


	for layer in reversed(range(len(W))):
		if layer==len(W)-1:
			delta[layer]=np.multiply(-(Yreal-Yth),dsigmoid(Z[layer]))

			gradient[layer]=np.dot(a[layer-1].transpose(),delta[layer])

		else:
			delta[layer]=np.dot(delta[layer+1],W[layer+1].transpose())*dsigmoid(Z[layer])
			gradient[layer]=np.dot(X.transpose(),delta[layer])

	return gradient

def gradient_descent(gradient,W,b,learning_rate):
	for layer in range(len(W)):
		W[layer]=W[layer]-learning_rate*gradient[layer]
	return W

#data retrieving------------------------------------------------------------------------------------
os.chdir("C:\\Users\\Nicolas\\Google Drive\\website\\python-own neuralnetwork")
data=pd.read_csv("data.csv",sep=";")

#m : number of exemples, p : numbers of features
m=data.shape[0]
p=data.shape[1]

#Separating Inputs and outputs ---------------------------------------------------------------------
X=data.ix[:,0:p-1] #not including p-1
y=data.iloc[:,p-1]
Yreal=[]
for i in range(len(y)):
	if y[i]==1:
		Yreal.append([1,0])
	elif y[i]==0:
		Yreal.append([0,1])

#parameters-----------------------------------------------------------------------------------------
n_neurons=[3]
n_hidden=len(n_neurons)

#normalization of the inputs and network initialization---------------------------------------------
X=normalize(X)
W,b=create_network(X,n_hidden,n_neurons,y)
	
#Calculate 1st forward propagation and initial cost-------------------------------------------------
Yth=forward(X,W,b)

J=cost(Yth,Yreal)

learning_rate=0.0005

#training the network-------------------------------------------------------------------------------
for epoch in range (10000):
	Yth=forward(X,W,b)
	J=cost(Yth,Yreal)
	print('cost is : ',J)
	grad=gradient(Yth,Yreal,X,J,W)
	for layer in range(len(W)):
		W[layer]=gradient_descent(grad[layer],W[layer],b[layer],learning_rate)