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
from sklearn import linear_model

def features_ext(cat,m10,m11,feature_words) :
    feature={}
    for w in feature_words :
        if w in cat :
            feature[w] = 1
        else:
            feature[w]=0
    if not np.isnan(m10):
        feature["m10"] = m10
    else:
        feature["m10"] = 0

    feature["m11"] = m11
    #print feature
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

dataset = []
##creating Trainig Data
#print np.isnan(month_m08[40])
for i in range(1,442):
    cat = json.loads(category[i])
    if not np.isnan(month_m08[i]):
        f= features_ext(cat,month_m10[i],month_m11[i],feature_words)
        f["VALUE"] = month_m08[i]
        dataset.append(f)
    else:
        f = features_ext(cat,month_m10[i],month_m11[i],feature_words)
        f["VALUE"] = -1
        dataset.append(f)

dataset = pd.DataFrame(dataset)
value = dataset["VALUE"]
del dataset["VALUE"]

#dataset.drop("VALUE")
train_data = dataset.ix[0:347,:]
test_data = dataset.ix[347:,:]
# train_data_frame
model = linear_model.LinearRegression()
model.fit(train_data,value.ix[0:347])
y=model.predict(test_data)
for i in range(347,len(month_m08)-1):
    month_m08.ix[i]= int(abs(y[i-347] ))
print month_m08