#parseF.py

import os
import sys
import csv
import pprint
import re
import nltk
from nltk.corpus import stopwords, words
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize



pp=pprint.PrettyPrinter(indent=4)
csv.field_size_limit(999999999)

stp = set(stopwords.words('english'))
wtf = set(words.words())
porter=nltk.PorterStemmer()
lmm = nltk.WordNetLemmatizer()

topLevel = os.path.dirname(__file__)


wcsv = open('8_Text_Normalized.csv','wt')

fnames = ['FilePath', 'From', 'Body', 'To', 'Date']
dwcsv = csv.DictWriter(wcsv,lineterminator='\n',fieldnames=fnames)
dwcsv.writeheader()

ct=0
mx=0
l=[]

bigDct={}


def rcontent(fl):
	#global bigDct
	fo=open(fl,'r')
	l={}
	cnt=0
	tmp=[]
	keyvar=''
	for i in fo:
		x=i.split(':')

		if(cnt==0):
			if(len(x)==1):
				l[keyvar]+=x[0]
			else:
				if(keyvar=='X-FileName'):
					continue
				l[x[0]]=':'.join(x[1:])

			if(x[0]=='X-FileName'):
				cnt=1
		else:
			tmp.append(i)

		if(len(x)>1):
			keyvar=x[0]
	
	s=' '.join(tmp)
	s=s.split('\n')
	s=' '.join(s)

	ss= s.split('Original Message')
	ss0 = ss[0].split('Forwarded by')
	res = ss0[0].strip('-')

	res = re.sub("[^A-Za-z]"," ",res)
	res = res.lower()
	resl = word_tokenize(res)
	
	fres = [i for i in resl if i not in stp]
	fresf = [i for i in fres if(i in wtf and len(i)>2)]

	fresfl = [porter.stem(i) for i in fresf]
	fresfinal = [lmm.lemmatize(i,'v') for i in fresfl]
	#return 
	l['Body']= ' '.join(list(set(fresfinal)))
	
	return l
	
def readFile(topLevel):
	global ct
	global dwcsv
	global bigDct
	#global mx
	#global l
	if(os.path.isfile(topLevel)):
		
		dct=rcontent(topLevel)
		
#		try:
		tmpfrm = dct['From'].strip() if 'From' in dct else dct['X-From'].strip()
		tmpto = dct['To'].strip() if 'To' in dct else dct['X-To'].strip()
		x = tmpfrm.split('@')
		bdy = dct['Body'].strip().split()
		#print(x)
		
		if(len(x)==2 and len(bdy)!=0):
			#print(' '.join(bdy))
			for i in set(bdy):
				if(i in bigDct):
					bigDct[i]+=1
				else:
					bigDct[i]=1
			if("enron.com" in x[1]):
				dwcsv.writerow({
					'FilePath':topLevel.strip(),
					'From':1,
					#'To':tmpto,
					#'Date':dct['Date'].strip(),
					'Body':' '.join(bdy)
				})
			else:
				dwcsv.writerow({
					'FilePath':topLevel.strip(),
					'From':0,
					#'To':tmpto,
					#'Date':dct['Date'].strip(),
					'Body':' '.join(bdy)
				})
		
		#except:
		#	print(topLevel)
		#	os._exit(1)
		
		ct+=1
		print(ct)
	
	else:
		for i in os.listdir(topLevel):
			readFile(topLevel+'/'+i)

readFile(topLevel)
print("Number of features: ",len(bigDct))

idfFile = open('IDF-CSV.csv','wt')
idff = csv.DictWriter(idfFile,lineterminator='\n',fieldnames=list(bigDct.keys()))
idff.writeheader()
idff.writerow(bigDct)


'''

#Reading a CSV file

f = < csv file name >

mode'rU' for reading in unicode
'wt' for writing
'a' append

with open(f,mode='rU',encoding='utf-8') as fileobj:
	# fpointer is the filepointer
	fpointer = csv.DictReader(fileobj)

	for i in fpointer:
		# do your thing here
		# i is a python dictionary with keys as fieldnames in the csv file
		# and values are the corresponding row it is pointing.
		# for loop increments the pointer to point to next row.

'''