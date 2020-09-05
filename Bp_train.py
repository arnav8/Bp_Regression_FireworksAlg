#encoding=utf-8
#Date 2017.03.10
#Three layer bp network
#The threshold is revised by the sum of each sample

from Fireworks import *
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

'''Neural network training function
parameter:
X: training data set, numpy array
Y: training label set, numpy array
h: the number of hidden neurons, integer
d: Whether to use FWALG algorithm for optimization, Boolean variable
return value:
The weights and thresholds of the hidden layer and output layer, numpy array
Iterative error of neural network, iterative error of firework algorithm, integer
'''
def MyBP(X, Y, h, d = False):
	#Boolean variable d indicates whether to use the firework algorithm to optimize the weights and thresholds
	if shape(X)[0] != shape(Y)[0]:
		print("The line of X and Y must be same!")

	m = shape(X)[0]; n = shape(X)[1]; l = shape(Y)[1]

	if d == True:
		#Input layer-hidden layer weight
		whj, vih, rh, thetaj, EE = FWA(X, Y, h)
		whj = array(whj); vih = array(vih); rh = array(rh); thetaj = array(thetaj)

	else:
		whj = random.random((n, h))
		vih = random.random((h, l))
		rh = random.random((1, h))
		thetaj = random.random((1, l))
		EE = 0

	a = [0] * 200

	for i in range(200):
		#Forward spread
                #Hidden layer input
		alphah = dot(X, whj)
		#Hidden layer output
		bh = sigmoid(alphah - rh)
		#Output layer input
		betaj = dot(bh, vih)
		#Output layer output
		ykj = sigmoid(betaj - thetaj)
		#error
		E = Y - ykj
		a[i] = sum(E * E) / 2

		#Backpropagation
		gj = sigmoid(ykj, True) * E
		#Hidden layer-output layer weight changes
		delta_vih = dot(bh.T, gj)

		#The threshold of the output layer is changed and corrected by all samples
		delta_thetaj = -gj[0, :]
		for i in range(1, m):
			delta_thetaj += -gj[i, :]

		#Input layer-hidden layer weight change
		delta_whj = dot(X.T, ((sigmoid(bh, True)) * dot(gj, vih.T)))

		#The hidden layer threshold is changed and corrected by all samples
		delta_rh1 = (sigmoid(bh, True)) * dot(gj, vih.T)
		delta_rh = -delta_rh1[0, :]
		for i in range(1, m):
			delta_rh += -delta_rh1[i, :]

		#Correction
		vih += delta_vih
		thetaj += delta_thetaj
		whj += delta_whj
		rh += delta_rh

	#The return value is the four parameters and the iteration error of the neural network and the iteration error of the firework algorithm
	return whj, rh, vih, thetaj, a, EE

'''Neural network prediction function
parameter:
X: Data set of the new sample, numpy array
whj: input layer-hidden layer weight
rh: hidden layer threshold
vih: hidden layer-output layer weight
thetaj: output layer threshold
return value:
Y: label of the new sample, numpy array
'''
def predictY(X, whj, rh, vih, thetaj):
	alphah = dot(X, whj)
	# In python, different rows of the same column of the array can be added and subtracted, and the ones with fewer rows can be supplemented with the previous row
	bh = sigmoid(alphah - rh)
	betaj = dot(bh, vih)
	Y = sigmoid(betaj - thetaj)
	return Y
