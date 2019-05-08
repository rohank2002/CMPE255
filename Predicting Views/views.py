import nltk
import numpy as np
import re
import string
from nltk.stem.lancaster import LancasterStemmer
import math
from scipy.sparse import csr_matrix
import re
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords #stack overflow
from nltk.tokenize import word_tokenize #geeks for geeks
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.linear_model import LinearRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from ml_metrics import rmse
us_videos = pd.read_csv('D:\CMPE255\CMPE 255 Project\Data/USvideos.csv')

#print(us_videos.head())
views=list(us_videos["views"])
description=list(us_videos["description"])
#print(views[0])

# Lower case
def lowercase(description):
    op = []
    for i in range(len(description)):
        l = str(description[i]).lower()
        op.append(l)
    return op
description=lowercase(description)
#remove numbers
def removeNums(description):
    result=[]
    for val in description:
        c=re.sub(r'\d+', '', val)
        result.append(c)
    return result
description=removeNums(description)
#print(description[0])
#import string
def removelinks(description):
    op =[]
    for val in description:
        val = re.sub('http[s]?://\S+', '', val)
        op.append(val)
    return op
descriptionwithoutlinks=removelinks(description)
stop_words = set(stopwords.words('english'))
def removepunctuations(string):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    no_punct = ""
    for char in string:
        if char not in punctuations:
            no_punct = no_punct + char
    return no_punct
o=[]
for val in descriptionwithoutlinks:
    c=removepunctuations(val)
    o.append(c)
descriptionwithoutlinks=o
#print(descriptionwithoutlinks[0])

def stemming(description):
    stemmer = PorterStemmer()
    op = []
    for val in description:
        val = word_tokenize(val)
        c = ""
        for word in val:
            a = stemmer.stem(word)
            c += a
            c += " "

        op.append(c)
    return op
c=stemming(descriptionwithoutlinks)
descriptionwithoutlinks=c
#print(descriptionwithoutlinks[0])

def tokenizer(descriptionwithoutlinks):
    stop_words = set(stopwords.words('english'))
    op=[]
    for val in descriptionwithoutlinks:
        token=word_tokenize(val)
        result = [i for i in token if not i in stop_words]
        op.append(result)
    return op

descriptionwithoutlinks=tokenizer(descriptionwithoutlinks)

#print(descriptionwithoutlinks[0])
#create matrix
indptr = [0]
indices = []
traindata = []
vocabulary = {}

for dec in descriptionwithoutlinks:
    for  words in dec:
        index = vocabulary.setdefault(words, len(vocabulary))
        indices.append(index)
        traindata.append(1)
    indptr.append(len(indices))
matrix = csr_matrix((traindata, indices, indptr), dtype=float).toarray()
length = len(matrix)

X_train, X_test, y_train, y_test = train_test_split(matrix, views, test_size=0.33, random_state=42)

#params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,'learning_rate': 0.01, 'loss': 'ls'}
#clf = ensemble.GradientBoostingRegressor(**params)
#clf.fit(X_train, y_train)

#y_predicted=clf.predict(X_test)
#print(len(y_test))
#print(len(y_predicted))
rf=LinearRegression()
rf.fit(X_train,y_train)
y_pred=rf.predict(X_test)
print(rmse(y_test,y_pred))