#email_label.py

import os
import sys
import csv
import pprint
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import codecs
from io import open

stp = set(stopwords.words('english'))
topLevel = os.path.dirname(__file__)+'4_stopword_tokenize_regex_EnronData_empty.csv'
print(topLevel)
csv.field_size_limit(999999999)
email_label={}
ct=0
mx=0
l=[]

wcsv = open('7_Uniq_Email.csv',mode='wt')#,encoding='utf-8')
cols = ['EmailID', 'Frequency', 'ID']
dwcsv = csv.DictWriter(wcsv,lineterminator='\n',fieldnames=cols)
dwcsv.writeheader()
def convert(topLevel):
   #global ct
   global email_label
   with codecs.open(topLevel,mode='rU') as fileobj:
      fpointer = csv.DictReader(fileobj)
      print(fpointer.fieldnames)
      ct=0
      for i in fpointer:
         #try:
         tmpfrm = i['From']
         #x = tmpfrm.split('@')
         #if( len(x)==2 and "enron.com" in x[1]):
         if(tmpfrm in email_label):
            email_label[tmpfrm]+=1
         else:
            email_label[tmpfrm]=1
         ct+=1
         print(ct,end='\r')
         #except:
         #   print('i')
         #   print(i)
      print(i)
convert(topLevel)
hshL = sorted(email_label.items(), key=lambda x: x[1])
index = 1
for i in hshL:
   try:
      dwcsv.writerow({
         'EmailID': i[0],
         'Frequency': i[1],
         'ID':index
      })
   except:
      print("Tatti")
   #print(index)
   index+=1
#dwcsv.writeheader()
#dwcsv.writerow(email_label)
