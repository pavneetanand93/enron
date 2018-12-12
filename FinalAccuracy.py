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

cntl=0
for i in range(len(label)):
	if(label[i]==prlabl[i]):
		cntl+=1

print("train accuracy LR: ",cntl/len(prlabl)) # output train error: 0.8148

###---------------------------------------------------------------------------###


## Naive Bayes
gnb = GaussianNB()
y_pred=gnb.fit(datasett,label).predict(datasett)
cntl=0
for i in range(len(label)):
	if(label[i]==y_pred[i]):
		cntl+=1

print("train accuracy NB: ",cntl/len(y_pred))
# output 0.5202

###----------------------------------------###

## SVM
from sklearn import svm
clfsvm = svm.SVC(gamma='scale')
prd = clf.fit(datasett,label)

trn_pred = prd.predict(datasett)

cntl=0
for i in range(len(label)):
	if(label[i]==trn_pred[i]):
		cntl+=1

print("Train error SVM: ",cntl/len(trn_pred))

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
#	print(ct)
#	ct+=1
cntl=0
for i in range(len(label)):
	if(label[i]==knn_prd[i]):
		cntl+=1	

print("3-NN accuracy: ",cntl/len(knn_prd))

#	acc.append(cntl/len(knn_prd))

# 
# 0.9286 for k=2
# 0.9373 for k=3

#plt.figure()
#plt.plot([i for i in range(1,200)], acc, linewidth=1, linestyle='solid')

###---------------------------------------###

# AdaBoost

from sklearn.ensemble import AdaBoostClassifier
cntl=0
clfada = AdaBoostClassifier(n_estimators=100)
prdada = clfada.fit(datasett,label)
ada_prd = prdada.predict(datasett)

for i in range(len(label)):
	if(label[i]==ada_prd[i]):
		cntl+=1

print("AdaBoost Accuracy: ",cntl/len(ada_prd))