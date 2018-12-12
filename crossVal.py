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
from sklearn.model_selection import cross_val_score
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


#datasett=np.array(datasett)
label = np.ravel(label)
print(datasett[0],label[0])

## LOGISTIC REGRESSION
clf = LogisticRegression(random_state=0, solver='lbfgs').fit(datasett, label)

prlabl=clf.predict(datasett)
scores = cross_val_score(clf, datasett, label, cv=5)
print(np.mean(scores)) # 0.8112

###---------------------------------------------------------------------------###


## Naive Bayes
gnb = GaussianNB()
y_pred=gnb.fit(datasett,label).predict(datasett)
scores = cross_val_score(gnb, datasett, label, cv=5)
print(np.mean(scores)) # 0.5514

###----------------------------------------###

## SVM
from sklearn import svm
clfsvm = svm.SVC()
prd = clfsvm.fit(datasett,label.astype(float))

trn_pred = prd.predict(datasett)
scores = cross_val_score(clfsvm, datasett, label, cv=5)
print(np.mean(scores)) # 0.8197

###----------------------------------------###

## K-NN
from sklearn.neighbors import KNeighborsClassifier
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt


#acc=[]
#ct=0
#for i in range(1,200):
neigh = KNeighborsClassifier(n_neighbors=3)
prdknn = neigh.fit(datasett,label)
knn_prd = prdknn.predict(datasett)
scores = cross_val_score(neigh, datasett, label, cv=5)
print(np.mean(scores)) # 0.6388
#	acc.append(cntl/len(knn_prd))

###---------------------------------------###

# AdaBoost

from sklearn.ensemble import AdaBoostClassifier
cntl=0
clfada = AdaBoostClassifier(n_estimators=100)
prdada = clfada.fit(datasett,label)
ada_prd = prdada.predict(datasett)
scores = cross_val_score(clfada, datasett, label, cv=5)
print(np.mean(scores)) # 0.7074
plt.figure()
plt.plot(scores, linewidth=1, linestyle='solid')
plt.show()