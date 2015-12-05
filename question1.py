''' Question 1 
MINEWHAT
Coded by kumar shubham
'''
import nltk 
import pandas as pd 
import numpy as np
import json 
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from collections import Counter

def feature_extractor(name_product):
	feature = {}
	name_words = tokenizer.tokenize(name_product.lower())
	name_word_modified = [t for t in name_words if name_words not in stopwords]
	feature["words"] = tuple(name_word_modified)
	return feature

tokenizer = RegexpTokenizer(r'\w+')
data = pd.read_csv("product.csv")



name_product = data["name"]
category = data["cat"]
category_count= []
stopwords = stopwords.words('english')
feature_set =[]

#print name_product
#trainig
for i in range(1,len(category)):
	name_product = data.ix[i,1]
	#name_words = tokenizer.tokenize(name_product.lower())
	temp = json.loads(category[i])
	category_count+=temp
	if  "uncategorised" not in temp:
		for t in temp:
			f=feature_extractor(name_product)
			feature_set.append((f,t))

classifier = nltk.NaiveBayesClassifier.train(feature_set)


#tessting

for i in range(1,len(category)):
	name_product = data.ix[i,1]
	#name_words = tokenizer.tokenize(name_product.lower())
	temp = json.loads(category[i])
	category_count+=temp
	if  "uncategorised" in temp:
		result = classifier.classify(feature_extractor(name_product))
		print name_product, result







#print feature_set
#count = Counter(category_count)

##creating the feature set


	