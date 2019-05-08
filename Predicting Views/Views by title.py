import nltk
import numpy as np
import re
import string
from nltk.stem.lancaster import LancasterStemmer
import math
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import re
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
from sklearn.linear_model import Lasso
from nltk.corpus import stopwords #stack overflow
from nltk.tokenize import word_tokenize #geeks for geeks
import pandas as pd
from sklearn.linear_model import Ridge
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import f1_score
from sklearn.linear_model import LinearRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from ml_metrics import rmse


us_videos = pd.read_csv('D:\CMPE255\CMPE 255 Project\Data/USvideos.csv')

#print(us_videos.head())
views=list(us_videos["views"])
#print(us_videos.head())
title=list(us_videos["title"])
#print(title.head())

#print(title[0])
def lowercase(description):
    op = []
    for i in range(len(description)):
        l = str(description[i]).lower()
        op.append(l)
    return op
title=lowercase(title)

#print(title[0])

def removeNums(description):
    result=[]
    for val in description:
        c=re.sub(r'\d+', '', val)
        result.append(c)
    return result
title=removeNums(title)


def removelinks(description):
    op =[]
    for val in description:
        val = re.sub('http[s]?://\S+', '', val)
        op.append(val)
    return op
titlewithoutlinks=removelinks(title)
stop_words = set(stopwords.words('english'))


def removepunctuations(string):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    no_punct = ""
    for char in string:
        if char not in punctuations:
            no_punct = no_punct + char
    return no_punct
o=[]
for val in titlewithoutlinks:
    c=removepunctuations(val)
    o.append(c)
titlewithoutlinks=o

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
c=stemming(titlewithoutlinks)
titlewithoutlinks=c



def tokenizer(descriptionwithoutlinks):
    stop_words = set(stopwords.words('english'))
    op=[]
    for val in descriptionwithoutlinks:
        token=word_tokenize(val)
        result = [i for i in token if not i in stop_words]
        op.append(result)
    return op



indptr = [0]
indices = []
traindata = []
vocabulary = {}

for dec in titlewithoutlinks:
    for  words in dec:
        index = vocabulary.setdefault(words, len(vocabulary))
        indices.append(index)
        traindata.append(1)
    indptr.append(len(indices))


matrix = csr_matrix((traindata, indices, indptr), dtype=float).toarray()
length = len(matrix)
#dim = TruncatedSVD(n_components=30)
#mat = dim.fit_transform(matrix)

X_train, X_test, y_train, y_test = train_test_split(matrix, views, test_size=0.33, random_state=42)

lr=LinearRegression()#RMSE 1.06321
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
print("RMSE BY LINEAR REGRESSION")
print(rmse(y_test,y_pred))
#print("R2 Score BY LINEAR REGRESSION")
#print(lr.score(y_test,y_pred) )
#clf = svm.SVC(gamma='scale')
#clf.fit(X_train,y_train)
#y_clf=clf.predict(X_test)
#print("RMSE BY SVM")
#print(rmse(y_test,y_clf))
#ridge=Ridge(alpha=0.1)
#ridge.fit(X_train,y_train)
#y_r=ridge.predict(X_test)

#print(rmse(y_test,y_r))
#lassoreg = Lasso(alpha=1,normalize=True, max_iter=1e5)
#lassoreg.fit(X_train,y_train)
#y_l=lassoreg.predict(X_test)
#print(rmse(y_test,y_l))

#regr = ElasticNet(random_state=12)
#regr.fit(X_train,y_train)
#y_el=regr.predict(X_test)
#print(rmse(y_test,y_el))
#clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)
#y=clf.predict(X_test)
#rint(rmse(y_test,y))
'''
rf = RandomForestRegressor(max_features=2, min_samples_split=4, n_estimators=50, min_samples_leaf=2)
gb = GradientBoostingRegressor(loss='quantile', learning_rate=0.0001, n_estimators=50, max_features='log2', min_samples_split=2, max_depth=1)
#ada_tree_backing = DecisionTreeRegressor(max_features='sqrt', splitter='random', min_samples_split=4, max_depth=3)
#ab = AdaBoostRegressor(ada_tree_backing, learning_rate=0.1, loss='square', n_estimators=1000)


rf.fit(X_train,y_train)
y_rf=rf.predict(X_test)
gb.fit(X_train,y_train)
y_gb=gb.predict(X_test)
print("RMSE BY RF")
print(rmse(y_test,y_rf))
print("RMSE BY GB ")
print(rmse(y_test,y_gb))


'''