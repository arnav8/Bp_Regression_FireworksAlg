#coding=utf-8
#Date 2017.03.12

from Bp_train import *
from numpy import *
import matplotlib.pyplot as plt

'''Neural network call function
parameter:
x: training set sample collection, numpy array
y: training set label collection, numpy array
newx: test set sample collection, numpy array
newy: test set label collection, numpy array
d: Boolean variable, whether to use the fireworks algorithm for optimization
return value:
train_error: training set error, floating point
error: test set error, floating point
'''
def BPneuralnetwork(x, y, newx, newy, d):
	#Normalization of training set
	xx = zeros([shape(x)[0], shape(x)[1]])
	yy = zeros([shape(y)[0], shape(y)[1]])

	for i in range(shape(x)[0]):
		for j in range(shape(x)[1]):
			xx[i, j] = (x[i, j] - min(x[:, j])) / (max(x[:, j] - min(x[:, j])))

	for i in range(shape(y)[0]):
		for j in range(shape(y)[1]):
			yy[i, j] = (y[i, j] - min(y[:, j])) / (max(y[:, j] - min(y[:, j])))

	#Seeking mean square error of training set
        #a is the iteration error of the neural network,
        #EE is the iteration error of the firework algorithm
	whj, rh, vih, thetaj, a, EE = MyBP(xx, yy, 6, d)
	alphah = dot(xx, whj)
	bh = sigmoid(alphah - rh)
	betaj = dot(bh, vih)
	NewY = sigmoid(betaj - thetaj)
	NewYY = zeros([shape(NewY)[0], shape(NewY)[1]])
	for i in range(shape(NewY)[0]):
		for j in range(shape(NewY)[1]):
			NewYY[i, j] = NewY[i, j] * (max(y[:, j] - min(y[:, j]))) + \
			                  min(y[:, j])

	train_error = sum((NewYY - y) * (NewYY - y)) / 2

	#Test set
	newxx = zeros([shape(newx)[0], shape(newx)[1]])
	#Test set normalization
	for i in range(shape(newx)[0]):
		for j in range(shape(newx)[1]):
			newxx[i, j] = (newx[i, j] - min(x[:, j])) / (max(x[:, j] - min(x[:, j])))

	#Neural network prediction
	predict_y = predictY(newxx, whj, rh, vih, thetaj)
	#Denormalization of predicted value
	newpredict_y = zeros([shape(predict_y)[0], shape(predict_y)[1]])
	for i in range(shape(predict_y)[0]):
		for j in range(shape(predict_y)[1]):
			newpredict_y[i, j] = predict_y[i, j] * (max(y[:, j] - min(y[:, j]))) + \
			                  min(y[:, j])

	#Find the mean square error of the test set
	error = sum((newpredict_y - newy) * (newpredict_y - newy)) / 2

	#plotting
        #Real value and predicted value image
	fig1 = plt.figure()
	ax = fig1.add_subplot(111)
																								#画点
	ax.scatter(NewYY.flatten(), y.flatten(), color = 'blue')
	ax.scatter(newpredict_y.flatten(), newy.flatten(), color = 'black')
	#Draw a point
	x = [0, 600]
	y = [0, 600]
	ax.plot(x, y, 'r')
	plt.title('BP neural network', fontname='times new Roman', fontsize='10.5')
	plt.xlabel('predictvalue', fontname='times new Roman', fontsize='10.5')
	plt.ylabel('realvalue', fontname='times new Roman', fontsize='10.5')
	plt.show()

	#Neural network error function image
	fig2 = plt.figure()
	bx = fig2.add_subplot(111)
	x = range(200)
	y = a
	bx.plot(x, y)
	plt.title('Error function', fontname='times new Roman', fontsize='10.5')
	plt.xlabel('Number of iterations', fontname='times new Roman', fontsize='10.5')
	plt.ylabel('Error', fontname='times new Roman', fontsize='10.5')
	plt.show()

	return train_error, error
