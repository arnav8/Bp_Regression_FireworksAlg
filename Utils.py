#encoding=utf-8
#Date 2017.5.19
#Fireworks Algorithm

from numpy import  *

'''sigmoid function
parameter:
x: argument, is a numpy array
d: Whether to perform derivative, it is a Boolean variable
return value:
The function value or the result of the derivation is a numpy array
'''
def sigmoid(x, d = False):
	if d == True:
		return x * (1 - x)
	else:
		return 1 / (1 + exp(-x))

'''Calculate fitness function
During the decoding process of individual fireworks, the weight and threshold parameters are obtained, and the fitness is obtained through these parameters, which is the mean square error in the neural network
parameter:
train_x: training set feature samples, numpy array
train_y: training set label sample, numpy array
x: current individual (obtained by encoding), numpy array
n: the number of neurons in the input layer, integer
h: the number of hidden neurons, integer
l: the number of neurons in the output layer, integer
return value:
Fitness function value (mean square error), floating point
'''
def calculatef(train_x,train_y,x,n,h,l):
	whj = zeros([n, h])
	vih = zeros([h, l])
	rh = zeros([1, h])
	thetaj = zeros([1, l])

	for i in range(n):
		for j in range(h):
			whj[i][j] = x[i*h+j]

	for i in range(h):
		for j in range(l):
			vih[i][j] = x[n*h+i*l+j]

	for i in range(h):
		rh[0][i] = x[n*h+h*l+i]

	for i in range(l):
		thetaj[0][i] = x[n*h+h*l+h+i]

	#Calculate fitness function (mean square error)
        # Hidden layer input
	alphah = dot(train_x, whj)
	# Hidden layer output
	bh = sigmoid(alphah - rh)
	# Output layer input
	betaj = dot(bh, vih)
	# Output layer output
	ykj = sigmoid(betaj - thetaj)
	# error
	E = train_y - ykj
	#Mean square error
	E = sum(E*E)/2
	return E

'''Decoding function
parameter:
x: current individual (obtained by encoding), numpy array
n: the number of neurons in the input layer, integer
h: the number of hidden neurons, integer
l: the number of neurons in the output layer, integer
E: mean square error
return value:
Input layer-hidden layer weight, hidden layer threshold, hidden layer-output layer weight, output layer threshold, all are numpy arrays
E: mean square error
'''
def final_weight(x, n, h, l, E):
	whj = zeros([n, h])
	vih = zeros([h, l])
	rh = zeros([1, h])
	thetaj = zeros([1, l])

	for i in range(n):
		for j in range(h):
			whj[i][j] = x[i*h+j]

	for i in range(h):
		for j in range(l):
			vih[i][j] = x[n*h+i*l+j]

	for i in range(h):
		rh[0][i] = x[n*h+h*l+i]

	for i in range(l):
		thetaj[0][i] = x[n*h+h*l+h+i]

	return whj, vih, rh, thetaj, E
