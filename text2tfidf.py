#parseF.py

import os
import sys
import csv
import pprint
import re
import nltk
import codecs
import sklearn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

csv.field_size_limit(999999999)
stp = set(stopwords.words('english'))

topLevel = os.path.dirname(__file__)


#with codecs.open()
txt = open('featureW.txt','wt')

with codecs.open('4_stopword_tokenize_regex_EnronData_empty.csv','rU') as ff:

	fptr=csv.DictReader(ff)
	bdy=''
	#dta=list()
	res=set()
	ct=0
	for i in fptr:
		print(ct)
		ct+=1

		dta = word_tokenize(i['Body'])
		#print(dta)
		for i in dta:
			if(len(i)>=3):
				res.add(i)

	for j in res:
		txt.write(j+',')
	print(len(res))





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