import numpy as np
import math
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import itertools
from numpy import genfromtxt


def getRFEst(n_est, max_feat):
	rf = RandomForestClassifier(n_estimators = n_est, max_features = max_feat)
	return rf

def getCV(estimator, data, labels):
	return np.mean(cross_validation.cross_val_score(estimator, data, labels, cv = 5))

def getBestRFParam(data, labels, n_est_Arr = [90,95,100,105,110], 
		max_feat_Arr=[490,495,500,505,510]):
	maxCV = float('-inf')
	maxParam = [1,1]
	params = []
	for n_est in n_est_Arr:
		for max_feat in max_feat_Arr:
			rf = getRFEst(n_est, max_feat)
			cv = getCV(rf, data, labels)
			if cv > maxCV:
				maxCV = cv
				maxParam = [n_est, max_feat]
			params.append([n_est,max_feat,cv])
	np.savetxt('rfZoomedParam.csv', params,fmt = '%d,%d,%f', delimiter = ',', newline = '\n')
	return maxParam

#Return SVM with c parameter.
def getSVMEst(c):
	return svm.LinearSVC(C = c)

#Find c between cmin and cmax that returns the min CV of an SVM model.
def getBestSVMParam(data, labels, cmin = 1, cmax = 50):
	maxCV = float('-inf')
	maxParam = 0
	params = []
	for c in np.arange(0.02,1,0.0005):
		svm = getSVMEst(c)
		cv = getCV(svm, data, labels)
		params.append([c,cv])
		if cv > maxCV:
			maxCV = cv
			maxParam = c
	np.savetxt('svmTinyParam.csv', params,fmt = '%f,%f', delimiter = ',', newline = '\n')
	return maxParam

#returns Kcluster estimator fitted on data.
def Kcluster(data):
	kmeans = KMeans(n_clusters = 5)
	kmeans.fit(data)
	return kmeans

#returns numpy array of indices corresponding to most important features.
def getImportantFeatInd(estimator, numFeats):
	try:
		importanceVector = np.array(estimator.feature_importances_)
	except AttributeError:
		importanceVector = np.array(estimator.coef_)
		importanceVector = abs(sum(importanceVector))
	return np.array(importanceVector.argsort()[-numFeats:][::-1])

def getAccTot(estimator, trainingData, trainingLabels):
	return estimator.score(trainingData, trainingLabels)

def getClassAcc(estimator, trainingData, trainingLabels):
	pred = estimator.predict(trainingData)
	cmat = confusion_matrix(trainingLabels, pred)
	return [(cmat[i][i])/float(sum(cmat[i])) for i in range(len(cmat))]

def clusterAcc(estimator, trainingData, trainingLabels):
	kmeansPred = estimator.predict(trainingData)
	permutations = [i for i in itertools.permutations([0,1,2,3,4])]
	maxAcc = 0
	for order in permutations:
		newLabels = [order[int(i-1)] for i in trainingLabels]
		newAcc = accuracy_score(newLabels, kmeansPred)
		if maxAcc < newAcc:
			maxAcc = newAcc
	return maxAcc


trainingData = genfromtxt('fm_final.csv', delimiter = ',')
trainingLabels = genfromtxt('training_labels.csv', delimiter = ',')
testData = genfromtxt('hrc_test_fm_final.csv', delimiter = ',')
#testLabels = 
featureArray = genfromtxt('fm_final_colnames.csv', delimiter =',', dtype= str)


#labels = np.genfromtxt('data.txt', delimiter=',', usecols=0, dtype=str)
#raw_data = np.genfromtxt('data.txt', delimiter=',')[:,1:]
#data = {label: row for label, row in zip(labels, raw_data)}



#rfParam = getBestRFParam(trainingData, trainingLabels)			#Get parameters of best RF model
#rfEst = getRFEst(rfParam[0], rfParam[1])						#Create RF estimator
#rfEst = getRFEst(110, 495)
#rfEst.fit(trainingData, trainingLabels)
#top_10_RF_Feat = featureArray[getImportantFeatInd(rfEst,10)]	#Get top 10 important features of RF model
#rfTotAcc = getAccTot(rfEst, trainingData, trainingLabels)
#rfClassAcc = getClassAcc(rfEst, trainingData, trainingLabels)


#svmParam = getBestSVMParam(trainingData, trainingLabels)
#svmEst = getSVMEst(svmParam)
#svmEst = getSVMEst(.015)
#svmEst.fit(trainingData, trainingLabels)
#top_10_SVM_Feat = featureArray[getImportantFeatInd(svmEst,10)]
#svmTotAcc = getAccTot(svmEst, trainingData, trainingLabels)
#svmClassAcc = getClassAcc(svmEst, trainingData, trainingLabels)

#rfEst = getRFEst(110, 495)						#Create RF estimator
#rfEst.fit(trainingData, trainingLabels)
#importantData = trainingData[:,getImportantFeatInd(rfEst,100)]
#kmeansEst = Kcluster(importantData)
#kmeansAcc = clusterAcc(kmeansEst, importantData, trainingLabels)

svmEst = getSVMEst(.015)
svmEst.fit(trainingData, trainingLabels)
testPred = svmEst.predict(testData)
testPred = np.array([int(i) for i in testPred])
np.savetxt('predict.txt', testPred,fmt = '%d', delimiter = '\n')