#coding=utf-8
#Date 2017.03.12

from numpy import *
import matplotlib.pyplot as plt

'''Linear regression function
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
def LinearRegression(x, y, newx, newy):

	xTx = x.T * x
	if linalg.det(xTx) == 0.0:
		print("This matrix is singular")
	ws = xTx.I * (x.T * y)
	yHat = x * ws

	#Calculate the training set r square
	SSE = (y - yHat).T * (y - yHat)
	error_train = SSE / 2
	#print 'Training set mean square error =',error_train
	yAvg = mean(y)
	SSR = (yHat - yAvg).T * (yHat - yAvg)
	SST = SSR + SSE
	rr = SSR / SST

	#Calculate the mean square error on the test set
	predict_y = newx * ws
	error = (predict_y - newy).T * (predict_y - newy) / 2
	#print '测试集均方误差 =', error

	#plotting
	fig = plt.figure()
	ax = fig.add_subplot(111)
	#plotting a point
	ax.scatter(yHat.flatten().A[0], y.flatten().A[0], color='blue')
	ax.scatter(predict_y.flatten().A[0], newy.flatten().A[0],color='black')
	#plot a line
	x = [0, 600]
	y = [0, 600]
	ax.plot(x, y, 'r')
	plt.title('linear regression', fontname='times new Roman', fontsize='10.5')
	plt.xlabel('predictvalue', fontname='times new Roman', fontsize='10.5')
	plt.ylabel('realvalue', fontname='times new Roman', fontsize='10.5')
	plt.show()

	return error_train, error, rr
