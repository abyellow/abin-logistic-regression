from numpy import *

xtest = loadtxt ('heartstatlog/heartstatlog_testSet.txt') 
xtest = xtest/amax(xtest,0)
ytest = loadtxt ('heartstatlog/heartstatlog_testLabels.txt') 
ytest = 2.*(ytest-1.5)

xset = loadtxt ('heartstatlog/heartstatlog_trainSet.txt')
xset = xset/amax(xset,0)
yset = loadtxt ('heartstatlog/heartstatlog_trainLabels.txt') 
yset = 2.*(yset -1.5)

#lmda = .01 
iterNum = 1000000
xveclen = size(xset,1)
wet_ini = array([0.1]*xveclen)
#wet_ini = loadtxt('wet_ini_lambda_'+str(lmda)+'.txt')

def rand_8020(xset,yset):

	xlen = size(xset,0)
	xtrainlen = int(xlen * .8)
	indices = random.permutation(xlen)
	train_idx, test_idx = indices[:xtrainlen], indices[xtrainlen:]
	xtrain, xtest = xset[train_idx,:], xset[test_idx,:]
	ytrain, ytest = yset[train_idx], yset[test_idx]

	return xtrain, xtest, ytrain, ytest	
	

def gred_method(wet_ini, lmda, xtrain, ytrain, iterNum):
	
	wet  = wet_ini
	xlen = size(xtrain,0)
	learningRate = .1

	for itr in range(1,iterNum):
	
		gradL = zeros(xveclen) + lmda*wet
		likehood = lmda/2.*dot(wet,wet)		

		for i in range(xlen):

			xdata = xtrain[i,:]
			ydata = ytrain[i]
			val = -ydata*dot(xdata,wet)
			if abs(val) > 700:
				val = sign(val)*700
			prob = 1/(1+exp(val))
			gradL += -ydata*xdata*(1-prob)
			likehood += log(1+exp(val))

		wet = wet - learningRate*gradL/sqrt(itr)#(linalg.norm(gradL)*sqrt(itr))

		check = linalg.norm(gradL)	
		if itr % 10000 == 0 :
			learningRate = sqrt(itr)*learningRate
		if itr % 100 == 0: 
			print 'iter_time:', itr, 'abs(gradL):', check, 'likehood:', likehood, 'lmda:', lmda
		if check <= 0.01:
			break
	print wet	
	savetxt('wet_ini_lambda_'+str(lmda)+'.txt',wet)
	return wet



def cross_valid(wet, xslt, yslt):
	xlen = size(xslt,0)
	xveclen = size(xslt,1)
	ypred = -1.*ones(xlen)
	
	for i in range(xlen):
		
		val = -1*dot(wet, xslt[i,:])
		prob = 1/(1+exp(val))
		if prob > .5:
			ypred[i] = 1	

	cross = sum(abs(ypred + yslt )) / (2.*xlen)

	return cross


def main(wet_ini, lmda, xset, yset, xtest, ytest, iterNum):

	#rand data set for 80% train data and 20% of slt data
	xtrain, xslt, ytrain, yslt = rand_8020(xset, yset)

	#gred to get wet of train data set
	wet = gred_method(wet_ini, lmda, xtrain, ytrain, iterNum)
	#wet = wet_ini
	
	#a. cross_train error
	cross_train = cross_valid(wet, xtrain, ytrain)
	print 'cross_train:', cross_train

	#b. average over cross_slt error
	cross_slt = 0
	avg_num = 100
	for i in range(avg_num):
		xtrain, xslt, ytrain, yslt = rand_8020(xset, yset)
		cross_slt += cross_valid(wet, xslt, yslt)
	cross_slt = cross_slt/avg_num
	print 'cross_slt:', cross_slt

	#c. cross_test error
	cross_test = cross_valid(wet, xtest, ytest)
	print 'cross_test:', cross_test

	#save cross_val
	cross = [cross_train, cross_slt, cross_test]
	savetxt('cross_val_lambda_'+str(lmda)+'.txt',cross)

all_lambda = [0.01, 0.05, 0.25, 1, 5, 25, 100]
for s in all_lambda:	
	main(wet_ini, s, xset, yset, xtest, ytest, iterNum)
