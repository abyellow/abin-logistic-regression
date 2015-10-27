#!/usr/bin/env python
"""
Edit Bin H.
machine learning -- logistic regression
"""
import numpy as np

#Class machine learning of logistic regression
class LogReg:

	#Initialize data set, iteration time, learning rate and weight function
	def __init__(self, xtest, ytest, xset, yset, iter_num=10000, wet_iv=.1,
			 learningRate = .1,lmda = .1):

		self.xtest,self.ytest = xtest, ytest
		self.xset,self.yset = xset, yset
		self.iter_num = iter_num
		self.xveclen = np.size(xset,1)
		self.wet_ini = np.array([wet_iv]*self.xveclen)
		self.learningRate = learningRate 
	
	#random distribute data set into 80% train data and 20% of test data
	def rand_8020(self, xset, yset):

		xlen = np.size(xset,0)
		xtrainlen = int(xlen * .8)
		indices = np.random.permutation(xlen)
		train_idx, test_idx = indices[:xtrainlen], indices[xtrainlen:]
		xtrain, xtest = xset[train_idx,:], xset[test_idx,:]
		ytrain, ytest = yset[train_idx], yset[test_idx]

		return xtrain, xtest, ytrain, ytest	
	
	#logistic regression calculation of weight function
	def learn(self, lmda, xtrain, ytrain):
		
		learningRate = self.learningRate 
		xveclen = self.xveclen
		
		wet  = self.wet_ini
		xlen = np.size(xtrain,0)

		for itr in range(1,self.iter_num):
		
			gradL = np.zeros(xveclen) + lmda*wet
			likehood = lmda/2.*np.dot(wet,wet)		

			for i in range(xlen):

				xdata = xtrain[i,:]
				ydata = ytrain[i]
				val = -ydata*np.dot(xdata,wet)
				if abs(val) > 700:
					val = sign(val)*700
				prob = 1/(1+np.exp(val))
				gradL += -ydata*xdata*(1-prob)
				likehood += np.log(1+np.exp(val))

			wet = wet - learningRate*gradL/np.sqrt(itr)

			check = np.linalg.norm(gradL)	
			if itr % 10000 == 0 :
				learningRate = np.sqrt(itr)*learningRate
			if itr % 100 == 0: 
				print 'iter_time:', itr, 'abs(gradL):', check, 'likehood:', likehood
			if check <= 0.01:
				break
		print 'wet:', wet, ' lambda:', lmda
		return wet


	#cross validation function to check overfitting
	def cross_valid(self, wet, xslt, yslt):

		xlen = np.size(xslt,0)
		xveclen = np.size(xslt,1)
		ypred = -1.*ones(xlen)
		
		for i in range(xlen):
			
			val = -1*np.dot(wet, xslt[i,:])
			prob = 1/(1+np.exp(val))
			if prob > .5:
				ypred[i] = 1	

		cross = sum(abs(ypred + yslt )) / (2.*xlen)
		return cross


if __name__ =='__main__':
	
	def main(lmda, xset, yset, xtest, ytest):

		#Class ml_rlr and input data sets
		lr1 = LogReg(xtest,ytest,xset,yset)

		#rand data set for 80% train data and 20% of slt data
		xtrain, xslt, ytrain, yslt = lr1.rand_8020(xset, yset)

		#gred to get wet of train data set
		wet = lr1.learn(lmda, xtrain, ytrain)
		
		#a. cross_train error
		cross_train = lr1.cross_valid(wet, xtrain, ytrain)
		print 'cross_train:', cross_train

		#b. average over cross_slt error
		cross_slt = 0
		avg_num = 100
		for i in range(avg_num):
			xtrain, xslt, ytrain, yslt = lr1.rand_8020(xset, yset)
			cross_slt += lr1.cross_valid(wet, xslt, yslt)
		cross_slt = cross_slt/avg_num
		print 'cross_slt:', cross_slt

		#c. cross_test error
		cross_test = lr1.cross_valid(wet, xtest, ytest)
		print 'cross_test:', cross_test

		#save cross_val
		cross = [cross_train, cross_slt, cross_test]
		savetxt('cp_cross_lambda/cross_val_lambda_'+str(lmda)+'.txt',cross)

	
	#Load train data and test data
	xtest = np.loadtxt ('heartstatlog/heartstatlog_testSet.txt') 
	ytest = np.loadtxt ('heartstatlog/heartstatlog_testLabels.txt') 
	xset = np.loadtxt ('heartstatlog/heartstatlog_trainSet.txt')
	yset = np.loadtxt ('heartstatlog/heartstatlog_trainLabels.txt') 

	#Renormalize of x data set and shift y data set to 1 and -1
	xtest = xtest/np.amax(xtest,0)
	ytest = 2.*(ytest-1.5)
	xset = xset/np.amax(xset,0)
	yset = 2.*(yset -1.5)
		
	#Given different lambda to test
	all_lambda = [0.01, 0.05, 0.25, 1, 5, 25, 100]
	for lda in all_lambda:	
		main(lda, xset, yset, xtest, ytest)
