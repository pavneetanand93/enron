#testTrain.py

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
import codecs

datasett =[]
label = []

wcsv = open('IDF-CSV.csv','rU')
wcc = csv.DictReader(wcsv)
dct={}
ct=0
for i in wcc:
	feati = list(i.keys())
	dct={feati[j]:j for j in range(len(feati))}

print(len(dct))



with open('7_FeatureData.csv','rU') as f:
	ff=csv.DictReader(f)
	ct1=0
	ct0=0
	cnt=0
	for i in ff:
		featarr = len(dct)*[0]

		featdat = i['feat'][1:-1].split(',')
		#print(featdat)
		if(i['From']=='1' and ct1<2600):
			for m in featdat:
				k=m.split(':')
				k[0]=k[0].strip()[1:-1]
				
				featarr[dct[k[0]]]=float(k[1])

			datasett.append(np.array(featarr))
			label.append(1)
			ct1+=1

		if(i['From']=='0' and ct0<2400):
			for n in featdat:
				k=n.split(':')
				k[0]=k[0].strip()[1:-1]
				
				featarr[dct[k[0]]]=float(k[1])

			datasett.append(np.array(featarr))
			label.append(0)
			ct0+=1
		cnt+=1
		if(ct0+ct1==5000):break
		#if(cnt==5000):break
		print(cnt)


datasett=np.array(datasett)
label = np.array(label)
print(datasett[0],label[0])

clf = LogisticRegression(random_state=0, solver='lbfgs').fit(datasett, label)

prlabl=clf.predict(datasett)

cntl=0
for i in range(len(label)):
	if(label[i]==prlabl[i]):
		cntl+=1

print("train accuracy: ",cntl/len(prlabl))
