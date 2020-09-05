#encoding=utf-8
#Date 2017.5.19
#Fireworks Algorithm

from Utils import *

'''Firework algorithm
The purpose is to get better initial values of neural network parameters through the firework algorithm, and then use gradient descent for iteration
parameter:
X: training set sample collection, numpy array
Y: training section label collection, numpy array
h: the number of hidden neurons
return value:
The optimized input layer-hidden layer weight, hidden layer threshold, hidden layer-output layer weight, and initial value of the output layer threshold are all numpy arrays
'''
def FWA(X, Y, h):
	if shape(X)[0] != shape(Y)[0]:
		print("The line of X and Y must be same!")
	m = shape(X)[0]; n = shape(X)[1]; l = shape(Y)[1]

	#Initialization algorithm parameters
        #Sparks total
	m = 50
		#Upper and lower limit
	a = 0.8; b = 0.04
		#Explosion amplitude
	A = 40
		#Number of fireworks Gaussian explosions
	mm = 5
		#Fireworks dimension
	dimension = n*h + h*l + h + l
		#Number of fireworks
	nn = 5
		#Maximum and minimum boundaries
	xmin = -5; xmax = 5

	#Initialize fireworks
	fireworks = zeros([nn, dimension])
	for i in range(nn):
		for j in range(dimension):
			fireworks[i][j] = random.uniform(-5, 5)

	#Initialize a new firework
	fireworks_new = zeros([nn, 100, dimension])

	#Initialize Gaussian spark
	fireworks_rbf = zeros([nn, dimension])

	#Sparks
        #The number of sparks produced by each firework
	Si = zeros([nn, 1])
		#Explosion radius of each firework
	Ai = zeros([nn, 1])
		#Spark limit
	si = zeros([nn, 1])
		#Calculate the fitness function value of each firework
	f = zeros([nn, 1])
		#Maximum and minimum fitness
	fmax = f[0]; fmin = f[nn-1]
		#Error function initialization
	E = zeros([5000, 1])


	#Firework algorithm iteration process
	for delta_num in range(5000):

		# Total number of sparks produced by ordinary explosions
		sum_new_fireworks = 0
		# Total fitness
		sum = 0
		#Calculate fitness and find the maximum and minimum
		for i in range(nn):
			f[i] = calculatef(X, Y, fireworks[i], n, h, l)
			if f[i] > fmax:
				fmax = f[i]
			if f[i] < fmin:
				fmin = f[i]
			sum += f[i]

			#Find the explosion radius and number of sparks for each firework
		for i in range(nn):
				#Calculate the number of sparks
			Si[i] = m * (fmax - f[i] + 0.0001) / (nn * fmax - sum + 0.0001)
			Si[i] = round(Si[i])
			if Si[i] < a * m:
				si[i] = round(a * m)
			elif Si[i] > b * m:
				si[i] = round(b * m)
			else:
				si[i] = round(Si[i])
				#Can not exceed the number of sparks limit
			if Si[i] > si[i]:
				Si[i] = si[i]

				#Calculate the total number of sparks produced by ordinary explosions
			sum_new_fireworks += int(Si[i])

			#Calculate the explosion radius
			Ai[i] = A * (f[i] - fmin + 0.0001) / (sum - nn * fmin + 0.0001)

				#Create a new spark
			for j in range(Si[i]):
					#Initialize a new spark
				fireworks_new[i][j] = fireworks[i]
					#Randomly select z dimensions
				z = random.randint(1, dimension)
					#Randomly select the first z
				zz = range(dimension)
				random.shuffle(zz)

					# Create a new spark
				for k in range(z):
					fireworks_new[i][j][zz[k]] += random.uniform(0, Ai[i])


		#Generate Gaussian sparks (each firework generates a Gaussian spark)
			# Randomly select z dimensions
		z = random.randint(1, dimension)
		zz = range(dimension)
		random.shuffle(zz)
			#Gaussian random number
		g = random.uniform(-1, 1)
			#Gaussian explosion operator
		for i in range(mm):
			for j in range(z):
				fireworks_rbf[i][zz[j]] = g * fireworks[i][zz[j]]


			#Construct total fireworks
		sum_fireworks = nn + sum_new_fireworks + mm
		fireworks_final = zeros([sum_fireworks, dimension])
		for i in range(nn):
			fireworks_final[i] = fireworks[i]

		for j in range(Si[0]):
			fireworks_final[nn+j] = fireworks_new[0][j]

		for i in range(nn-1):
			for j in range(Si[i+1]):
				#print 'Si = ',Si[i]
				fireworks_final[int(nn+j+Si[i])] = fireworks_new[i+1][j]

		for i in range(mm):
			fireworks_final[int(nn+sum_new_fireworks+i)] = fireworks_rbf[i]


		#Mapping rule
		for i in range(sum_fireworks):
			for j in range(dimension):
				if fireworks_final[i][j] > xmax or fireworks_final[i][j] < xmin:
					fireworks_final[i][j] = xmin + mod(abs(fireworks_final[i][j]), \
				                           (xmax - xmin))

		#Choose a strategy
                #New population fitness after explosion
		f_new = zeros([sum_fireworks, 1])
		f_new_min = f_new[0]
		#print f_new_min
			#Initialize the optimal fitness index
		min_i = 0
			#Select the next generation of n individuals, consisting of the maximum fitness individual and nn-1 individuals farther away
			 #Find the optimal fitness
		for i in range(sum_fireworks):
			#print fireworks_final[i]
			f_new[i] = calculatef(X, Y, fireworks_final[i], n, h, l)
			if f_new[i] < f_new_min:
				f_new_min = f_new[i]
				min_i = i


			#Find the probability of each individual being selected
                         #Initialize the distance between two bodies
		D = zeros([sum_fireworks, sum_fireworks])
			#Calculate the distance between two bodies
		for i in range(sum_fireworks):
			for j in range(sum_fireworks):
				D[i][j] = dot((fireworks_final[i] - fireworks_final[j]), \
				              (fireworks_final[i] - fireworks_final[j])) / 2

			#Initialize the sum of the distance between each individual and other individuals
		Ri = zeros([sum_fireworks, 1])
			#Initialize a copy of the distance matrix
		RRi = zeros([sum_fireworks, 1])
			#Calculate the sum of the distance between each individual and other individuals
		for i in range(sum_fireworks):
			for j in range(sum_fireworks):
				Ri[i] += D[i][j]
		RRi = Ri

			#Select nn-1 individuals with the farthest distance, that is, sort the distance matrix
		for i in range(sum_fireworks-1):
			for j in range(i, sum_fireworks):
				if Ri[i] < Ri[j]:
					tmp = Ri[i]
					Ri[i] = Ri[j]
					Ri[j] = tmp

			#Construct a new population
		fireworks[0] = fireworks_final[min_i]
		for i in range(sum_fireworks):
			if Ri[0] == RRi[i]:
				fireworks[1] = fireworks_final[i]
			if Ri[1] == RRi[i]:
				fireworks[2] = fireworks_final[i]
			if Ri[2] == RRi[i]:
				fireworks[3] = fireworks_final[i]
			if Ri[3] == RRi[i]:
				fireworks[4] = fireworks_final[i]

			#After iteration, return the best individual
                     #Initialize the optimal fitness index
		ii = 0
		for i in range(nn):
			f[i] = calculatef(X, Y, fireworks[i], n, h, l)
			fmin = f[0]
			if f[i] < fmin:
				ii = i

		E[delta_num] = f[ii]

	return final_weight(fireworks[ii], n, h, l, E)
