#classificationF.py

import os
import sys
import csv
import pprint
import re
import numpy as np
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import IncrementalPCA
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature
from sklearn.metrics import average_precision_score
import codecs

datasett=[]
label=[]

with open('redPCA.csv','rU',encoding='utf-8') as dat, open('labels.csv','rU',encoding='utf-8') as labl:
	dat1 = csv.reader(dat)
	for i in dat1:
		ll = np.array([float(j) for j in i])
		datasett.append(ll)
	datasett=np.array(datasett)

	labl1 = csv.reader(labl)
	for i in labl1:
		dd = np.array([int(j) for j in i])
		label.append(dd)
	label=np.array(label)
	print(label.shape)


with open('redPCATesting.csv','rU',encoding='utf-8') as dat, open('labelsTesting.csv','rU',encoding='utf-8') as labl:

	dat1 = csv.reader(dat)
	labl1 = csv.reader(labl)
	truLabl = []
	for j in labl1:
		dd = np.array(int(j[0]))
		truLabl.append(dd)

	#datasett=np.array(datasett)
	label = np.ravel(label)
	#print(datasett[0],label[0])

	lrErr = []
	svmErr = []
	nbErr = []
	knnErr = []
	ensmErr = []

	clf = LogisticRegression(random_state=0, solver='lbfgs').fit(datasett, label)
	
	gnb = GaussianNB()
	nbOb=gnb.fit(datasett,label)

	from sklearn import svm
	clfsvm = svm.SVC()
	prd = clfsvm.fit(datasett,label)

	from sklearn.neighbors import KNeighborsClassifier
	import matplotlib
	matplotlib.use('Qt4Agg')
	import matplotlib.pyplot as plt
	neigh = KNeighborsClassifier(n_neighbors=3)
	prdknn = neigh.fit(datasett,label)

	from sklearn.ensemble import AdaBoostClassifier
	clfada = AdaBoostClassifier(n_estimators=100)
	prdada = clfada.fit(datasett,label)

	cnt=0
	for i in dat1:
		cnt+=1
		ll = np.array([float(j) for j in i])
	
		## LOGISTIC REGRESSION
		
		prlabl=clf.predict(np.array([ll]))
		lrErr.append(prlabl[0])

		###---------------------------------------------------------------------------###

		## Naive Bayes
		y_pred=nbOb.predict(np.array([ll]))
		nbErr.append(y_pred[0])

		###----------------------------------------###

		## SVM
		trn_pred = prd.predict(np.array([ll]))
		svmErr.append(trn_pred[0])

		###----------------------------------------###

		## K-NN
		knn_prd = prdknn.predict(np.array([ll]))
		knnErr.append(knn_prd[0])

		###---------------------------------------###

		# AdaBoost
		ada_prd = prdada.predict(np.array([ll]))
		ensmErr.append(ada_prd[0])
		print(cnt)


	#LR Test ERROR: 0.8702 
	cntl=0
	for i in range(len(truLabl)):
		if(truLabl[i]==lrErr[i]):
			cntl+=1
	print("test accuracy LR: ",cntl/len(lrErr))
	precision, recall, _ = precision_recall_curve(lrErr, truLabl)
	# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
	step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
	average_precision = average_precision_score(lrErr, truLabl)

	plt.figure()
	plt.step(recall, precision, color='b', alpha=0.2,
	         where='post')
	plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.0])
	plt.title('2-class Precision-Recall curve LR: AP={0:0.2f}'.format(
	          average_precision))
	plt.show()


	#NB error: 0.5928
	cntl=0
	for i in range(len(truLabl)):
		if(truLabl[i]==nbErr[i]):
			cntl+=1
	print("test accuracy NB: ",cntl/len(nbErr))	
	precision, recall, _ = precision_recall_curve(nbErr, truLabl)
	# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
	step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
	average_precision = average_precision_score(nbErr, truLabl)

	plt.figure()
	plt.step(recall, precision, color='b', alpha=0.2,
	         where='post')
	plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.0])
	plt.title('2-class Precision-Recall curve NB: AP={0:0.2f}'.format(
	          average_precision))
	plt.show()

	#SVM Error: 0.8702
	cntl=0
	for i in range(len(truLabl)):
		if(truLabl[i]==lrErr[i]):
			cntl+=1
	print("test accuracy SVM: ",cntl/len(svmErr))
	precision, recall, _ = precision_recall_curve(svmErr, truLabl)
	# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
	step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
	average_precision = average_precision_score(svmErr, truLabl)

	plt.figure()
	plt.step(recall, precision, color='b', alpha=0.2,
	         where='post')
	plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.0])
	plt.title('2-class Precision-Recall curve SVM: AP={0:0.2f}'.format(
	          average_precision))
	plt.show()

	#3-NN error: 0.6977 
	cntl=0
	for i in range(len(truLabl)):
		if(truLabl[i]==knnErr[i]):
			cntl+=1
	print("test accuracy 3-NN: ",cntl/len(knnErr))
	precision, recall, _ = precision_recall_curve(knnErr, truLabl)
	# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
	step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
	average_precision = average_precision_score(knnErr, truLabl)
	plt.figure()
	plt.step(recall, precision, color='b', alpha=0.2,
	         where='post')
	plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.0])
	plt.title('2-class Precision-Recall curve 3-NN: AP={0:0.2f}'.format(
	          average_precision))
	plt.show()

	#Adaboost Error: 0.8536
	cntl=0
	for i in range(len(truLabl)):
		if(truLabl[i]==ensmErr[i]):
			cntl+=1
	print("test accuracy Adaboost: ",cntl/len(ensmErr))
	precision, recall, _ = precision_recall_curve(ensmErr, truLabl)
	# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
	step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
	average_precision = average_precision_score(ensmErr, truLabl)

	plt.figure()
	plt.step(recall, precision, color='b', alpha=0.2,
	         where='post')
	plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.0])
	plt.title('2-class Precision-Recall curve Adaboost: AP={0:0.2f}'.format(
	          average_precision))
	plt.show()

