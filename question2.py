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
import numpy as np 
import sklearn 

def features_ext(cat,m09,m10,m11,feature_words) :
    feature={}
    for w in feature_words :
        if w in cat :
            feature[w] = 1
        else:
            feature[w]=0
    feature["m09"]= m09
    feature["m10"] = m10
    feature["m11"] = m11

    return feature





data = pd.read_csv("product.csv")
name_product = data["name"]
category = data["cat"]
month_m08 = data.ix[:,5]
month_m09 = data.ix[:,6]
month_m10 = data.ix[:,7]
month_m11 = data.ix[:,8]

feature_set = {}
category_count= []

for i in range(1,len(category)):
    #name_product = data.ix[i,1]
    #name_words = tokenizer.tokenize(name_product.lower())
    temp = json.loads(category[i])
    category_count+=temp

##extracting dominate words 
temp = Counter(category_count)

feature_words = [t for t in temp.keys() if temp[t]>5]

train_dataset = []
##creating Trainig Data
for i in range(1,346):
    cat = json.loads(category[i])
    if month_m08[i] is not "NAN":
        train_dataset.append((tuple(features_ext(cat,month_m09[i],month_m10[i],month_m11[i],feature_words)),month_m08[i]))
