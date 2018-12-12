#parseF.py

import os
import sys
import csv
import pprint
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


stp = set(stopwords.words('english'))

topLevel = os.path.dirname(__file__)


email_label={}
ct=0
mx=0
lbl=1
l=[]

def rcontent(fl):
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
	'''
	s='\n'.join(tmp)
	s=s.split('\n')
	s=''.join(s)

	ss= s.split('Original Message')
	ss0 = ss[0].split('Forwarded by')
	res = ss0[0].strip('-')

	res = re.sub("[^0-9A-Za-z$]"," ",res)
	res = res.lower()
	resl = word_tokenize(res)
	fres = [i for i in resl if i not in stp]

	l['Body']=' '.join(fres)
	'''
	return l
	
def readFile(topLevel):
	global ct
	global dwcsv
	global lbl
	global email_label

	if(os.path.isfile(topLevel)):
		dct=rcontent(topLevel)
		
#		try:
		tmpfrm = dct['From'].strip() if 'From' in dct else dct['X-From'].strip()
		x = tmpfrm.split('@')

		if(len(x)==2 and "enron.com" in x[1]):
			if(tmpfrm not in email_label):
				email_label[tmpfrm]=lbl
				lbl+=1
	
#		except:
#			print(topLevel)
		#	os._exit(1)
		
		ct+=1
		print(ct)
	
	else:
		for i in os.listdir(topLevel):
			readFile(topLevel+'/'+i)

readFile(topLevel)

wcsv = open('7_Uniq_Email.csv','wt')
dwcsv = csv.DictWriter(wcsv,lineterminator='\n',fieldnames=list(email_label.keys()))
dwcsv.writeheader()

dwcsv.writerow(email_label)




