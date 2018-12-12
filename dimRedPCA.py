#dimRedPCA.py

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

redf = open('redPCATesting.csv','a')
lablf = open('labelsTesting.csv','a')
rf = csv.writer(redf,lineterminator='\n')
lf = csv.writer(lablf,lineterminator='\n')

wcsv = open('IDF-CSV.csv','rU')
wcc = csv.DictReader(wcsv)
dct={}
ct=0
for i in wcc:
	feati = list(i.keys())
	dct={feati[j]:j for j in range(len(feati))}

print(len(dct))

transformer = IncrementalPCA(n_components=200, batch_size=200)

with open('7_FeatureData.csv','rU') as f:
	ff=csv.DictReader(f)
	cnt=0

	for i in ff:
		cnt+=1
		if(cnt==15000):break

	for i in ff:
		featarr = len(dct)*[0]

		featdat = i['feat'][1:-1].split(',')
		
		for m in featdat:
			k=m.split(':')
			k[0]=k[0].strip()[1:-1]
			
			featarr[dct[k[0]]]=float(k[1])

		datasett.append(np.array(featarr))
		label.append(i['From'])

		if(len(datasett)==1000):
			datasett=np.array(datasett)
			print(datasett.shape)
			dat_red=transformer.fit_transform(datasett)

			rf.writerows(dat_red)
			lf.writerows(label)

			print(dat_red.shape)
			print(dat_red)
			datasett=[]
			label=[]
			#break

		cnt+=1
		print(cnt)



'''
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
'''