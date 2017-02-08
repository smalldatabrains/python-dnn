#dependencies

import numpy as np #matrix multiplication + maths
import pandas #data management
import os

#data retrieving
os.chdir("C:\\Users\\Nicolas\\Desktop\\Company\\python-own neuralnetwork")
data=pandas.read_csv("data.csv",sep=";")

#m : number of exemples, p : numbers of features
m=data.shape[0]
p=data.shape[1]

#Separating Inputs and outputs
X=data.ix[:,0:p-1] #not including p-1
y=data.iloc[:,p-1]

#seed random number
np.random.seed(0)

# Z : output vector for each layer
# X : input vector for each layer
# W : Weights matrix
# a : values vector after applying the sigmoid function to Z, a becomes the new input vector for the next layer
# y : real value vector of the ouput
# yth : theoritical value vector calculated by our network
# e : error vector between y and yth
# J : costfunction, return single value

#normalization of the inputs	
def scaling(X):
	Xnorm=X/np.amax(X)
	return Xnorm
	
Xnorm=scaling(X)

class myneuralnetwork(X,y):
	#hyperparameters and weights
	def __init__(self):
		#variables
		self.learning_rate=0.01
		self.regularizationterm=2
		self.n_sample=1000
		self.n_hidden=2
		self.n_neurons=[p-1,5,3,1] #for each hidden layer we set the number of neurons, p-1 refers to input, last layer is the output y
		
		self.W={}
		#initialize random weights for first forward propagation (include bias neurons for each layer)
		for layer in range(n_hidden+1):
			self.W[layer]=np.random.randn(n_neurons[layer],n_neurons[layer+1]+1)

		
	#sigmoid function definition
	def sigmoid(z):
		sigmoid=1.0/(1.0+np.exp(-z))
		return sigmoid
		
	def dsigmoid(z):
		return np.exp(-z)/(1+np.exp(-z))^2
	
	#forward propagation
	def forward(self,X):
		self.Z={}
		self.A={}
		self.A[0]=X
		for layer in range(n_hidden+1):
			self.Z[layer+1]=np.dot(self.A[layer],self.W[layer])
			A[layer+1]=self.sigmoid(self.Z[layer+1])
	
		yth=A[self.n_hidden+1]
		return yth
	
	#cost function and gradient calculation
	def cost():
		J=np.sum(1/2*(y-yth)^2)
		return J
	
	def dcost():
		dJ={}
		for layer in range(n_hidden+1):
			dJ[layer]=1
		return dJ
	
myneuralnetwork(Xnorm,y)