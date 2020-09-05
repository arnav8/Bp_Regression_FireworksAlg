#coding=utf-8
#Date 2017.03.12
#Graduation project realization

from numpy import *
import matplotlib.pyplot as plt

'''Nonlinear regression function
parameter:
x: training set sample collection, numpy array
y: training set label collection, numpy array
newx: test set sample collection, numpy array
newy: test set label collection, numpy array
return value:
train_error: training set error, floating point
error: test set error, floating point
rr: regression training set r square
'''
def NonLinearRegression(x, y, newx, newy):
	xMat = x
	yMat = log(y)
	xTx = xMat.T * xMat
	if linalg.det(xTx) == 0.0:
		print("This matrix is singular")
	ws = xTx.I * (xMat.T * yMat)
	yHat = xMat * ws
	yHat = exp(yHat)

	#Calculate the training set r square
	SSE = (y - yHat).T * (y - yHat)
	error_train = SSE / 2
	yAvg = mean(y)
	SSR = (yHat - yAvg).T * (yHat - yAvg)
	SST = SSR + SSE
	rr = SSR / SST

	#Calculate the mean square error on the test set
	predict_y = newx * ws
	predict_y = exp(predict_y)
	error = (predict_y - newy).T * (predict_y - newy) / 2

	#plot
	fig = plt.figure()
	ax = fig.add_subplot(111)
	#plot a point
	ax.scatter(predict_y.flatten().A[0], newy.flatten().A[0], color = 'black')
	ax.scatter(yHat.flatten().A[0], y.flatten().A[0], color = 'blue')
	#plot a line
	x = [0, 600]
	y = [0, 600]
	ax.plot(x, y, 'r')
	plt.title('nonlinear regression', fontname='times new Roman', fontsize='10.5')
	plt.xlabel('predictvalue', fontname='times new Roman', fontsize='10.5')
	plt.ylabel('realvalue', fontname='times new Roman', fontsize='10.5')
	plt.show()

	return error_train, error, rr
