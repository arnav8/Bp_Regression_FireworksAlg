#coding=utf-8
#Date 2017.06.05

from LinearRegression import *
from BPnn import *
from NonLinearRegression import *
from BP_NonLinearRegression import *

'''Load data set from local
parameter:
fileName: data set name, string, if not in the current path, it should be an absolute path
return value:
dataMat: training set feature set, list type
labelMat: training set label collection, list type
'''
def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1]), float(lineArr[2]), \
                        float(lineArr[3]), float(lineArr[4])])
        labelMat.append(float(lineArr[5]))
    return dataMat, labelMat

print 'Please select the modeling method to be used：'
print '1: linear regression\n','2:nonlinear regression\n','3:neural network\n','4:nonlinear regression+neural network\n',\
'5: Fireworks algorithm optimizes neural network\n'
a = raw_input("Enter your input: ")
if a == '1':
	#n times k-fold cross validation
	dataMat, labelMat = loadDataSet('9.txt')
	dataMat = mat(dataMat); labelMat = mat(labelMat)
	labelMat = labelMat.T
	sum_error_train = 0; sum_error = 0; sum_rr = 0
	for j in range(10):
		z = range(27)
		random.shuffle(z)
		x = zeros([20, 5]); y = zeros([20, 1]); newx = zeros([7, 5]); newy = zeros([7, 1])
		x = mat(x); y = mat(y); newx = mat(newx); newy = mat(newy)
		for i in range(20):
			x[i] = dataMat[z[i]]
			y[i] = labelMat[z[i]]
		for i in range(20, 27):
			newx[i-20] = dataMat[z[i]]
			newy[i-20] = labelMat[z[i]]
		# Call the linear regression method to get the error and r square of the training set and test set
		error_train, error, rr = LinearRegression(x, y, newx, newy)
		sum_error_train += error_train; sum_error += error; sum_rr += rr
	error_train = sum_error_train / 10; error = sum_error / 10; rr = sum_rr / 10
	print 'Training set mean square error=', error_train
	print 'Validation set mean square error=', error
	print 'r square=', rr

elif a == '2':
	#n times k-fold cross validation
	dataMat, labelMat = loadDataSet('10.txt')
	dataMat = mat(dataMat); labelMat = mat(labelMat)
	labelMat = labelMat.T
	sum_error_train = 0; sum_error = 0; sum_rr = 0
	for j in range(10):
		z = range(27)
		random.shuffle(z)
		x = zeros([20, 5]); y = zeros([20, 1]); newx = zeros([7, 5]); newy = zeros([7, 1])
		x = mat(x); y = mat(y); newx = mat(newx); newy = mat(newy)
		for i in range(20):
			x[i] = dataMat[z[i]]
			y[i] = labelMat[z[i]]
		for i in range(20, 27):
			newx[i-20] = dataMat[z[i]]
			newy[i-20] = labelMat[z[i]]
		# # Call the nonlinear regression method to get the error and r-square of the training set and test set
		error_train, error, rr = NonLinearRegression(x, y, newx, newy)
		sum_error_train += error_train; sum_error += error; sum_rr += rr
	error_train = sum_error_train / 10; error = sum_error / 10; rr = sum_rr / 10
	print 'Training set mean square error=', error_train
	print 'Validation set mean square error=', error
	print 'r square=', rr

elif a == '3':
	x, y = loadDataSet('1.txt')
	newx, newy = loadDataSet('2.txt')
	x = array(x); y = array([y]); newx = array(newx); newy = array([newy])
	y = y.T; newy = newy.T
	xx = zeros([shape(x)[0], shape(x)[1]-1])
	newxx = zeros([shape(newx)[0], shape(newx)[1]-1])
	for i in range(shape(x)[0]):
		for j in range(1, shape(x)[1]):
			xx[i][j-1] = x[i][j]
	for i in range(shape(newx)[0]):
		for j in range(1, shape(newx)[1]):
			newxx[i][j-1] = newx[i][j]
	# Call the neural network method to get the error of the training set and test set
	train_error, error = BPneuralnetwork(xx, y, newxx, newy, False)
	print 'Training set mean square error=', train_error
	print 'Validation set mean square error=', error

elif a == '4':
	x, y = loadDataSet('3.txt')
	newx, newy = loadDataSet('4.txt')
	x = array(x); y = array([y]); newx = array(newx); newy = array([newy])
	y = y.T; newy = newy.T
	xx = zeros([shape(x)[0], shape(x)[1]-1])
	newxx = zeros([shape(newx)[0], shape(newx)[1]-1])
	for i in range(shape(x)[0]):
		for j in range(1, shape(x)[1]):
			xx[i][j-1] = x[i][j]
	for i in range(shape(newx)[0]):
		for j in range(1, shape(newx)[1]):
			newxx[i][j-1] = newx[i][j]
	# Call the combination method of neural network and nonlinear regression (without using intelligent optimization algorithm) to get the error of training set and test set
	train_error, error = BP_NonLinearRegression(xx, y, newxx, newy, False)
	print 'Training set mean square error=', train_error
	print 'Validation set mean square error=', error

else:
	x, y = loadDataSet('3.txt')
	newx, newy = loadDataSet('4.txt')
	x = array(x); y = array([y]); newx = array(newx); newy = array([newy])
	y = y.T; newy = newy.T
	xx = zeros([shape(x)[0], shape(x)[1]-1])
	newxx = zeros([shape(newx)[0], shape(newx)[1]-1])
	for i in range(shape(x)[0]):
		for j in range(1, shape(x)[1]):
			xx[i][j-1] = x[i][j]
	for i in range(shape(newx)[0]):
		for j in range(1, shape(newx)[1]):
			newxx[i][j-1] = newx[i][j]
	#Call the combination method of neural network and nonlinear regression (using intelligent optimization algorithm) to get the error of training set and test set
	train_error, error = BP_NonLinearRegression(xx, y, newxx, newy, True)
	print 'Training set mean square error=', train_error
	print 'Validation set mean square error=', error
