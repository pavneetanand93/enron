#dictFeat.py

import os
import sys
import csv
import pprint
import re
import nltk
import codecs
import math
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

pp=pprint.PrettyPrinter(indent=4)
csv.field_size_limit(999999999)
stp = set(stopwords.words('english'))

topLevel = os.path.dirname(__file__)


#with codecs.open()
#txt = open('featureW.txt','rU')
writeFeat = open('7_FeatureData.csv','wt')
fld = ['From','feat']
wft= csv.DictWriter(writeFeat,lineterminator='\n',fieldnames=fld)
wft.writeheader()

bigDct={}
idfFile = open('IDF-CSV.csv','rU')
idfs = csv.DictReader(idfFile)
for i in idfs:
	bigDct=i
print(len(bigDct))
idfFile.close()


with codecs.open('8_Text_Normalized.csv','rU') as ff:

	fptr=csv.DictReader(ff)

	
	ct=0
	for k in fptr:
		print(ct)
		res={}
		ct+=1

		dta = word_tokenize(k['Body'])
		
		for i in dta:
			if(i not in res):
				res[i]=1
			else:
				res[i]+=1

		for i in res:
			res[i]=(res[i]/len(dta))*math.log(461002/int(bigDct[i]))

		wft.writerow({
			'From':k['From'],
			'feat':res
			})
		#pp.pprint(res)
		#if(ct==30):break
		print(ct)
	
	#for j in res:
	#	txt.write(j+',')
	#print(len(res))





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