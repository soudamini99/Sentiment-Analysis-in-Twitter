import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB 
from sklearn.metrics import roc_auc_score
import  csv, re, json
import preprocess_taskCE
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import f1_score
from sklearn.datasets import make_classification

df= pd.read_csv("data24C.txt", sep='\t', names=['tweetid','topic','sentiment','tweet'])


stopset = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(use_idf=True,lowercase=True, strip_accents='ascii', stop_words=stopset)

y= df.sentiment

d_y = y



x= vectorizer.fit_transform(df.tweet)

print (y.shape)
print (x.shape)


trainattributes, testattributes, trainclasses, testclasses = train_test_split (x, y, random_state=0 )

noofposneg = testclasses
df = pd.DataFrame({'0':noofposneg})
qwe1 = df['0'].value_counts()
ss = qwe1.loc[0]
ww = qwe1.loc[1]
xx = qwe1.loc[2]
yy = qwe1.loc[-1]
zz = qwe1.loc[-2]
l = len(testclasses)
#print(testattributes)
p1 = ss/(ss+ww+xx+yy+zz)
p2 = ww/(ss+ww+xx+yy+zz)
p3 = xx/(ss+ww+xx+yy+zz)
p4 = yy/(ss+ww+xx+yy+zz)
p5 = zz/(ss+ww+xx+yy+zz)


#print(trainattributes)

cl_NB = MultinomialNB().fit(trainattributes, trainclasses)

y_predict1 = cl_NB.predict(testattributes)


cl_LR = linear_model.LogisticRegression(C=1e5)

cl_LR.fit(trainattributes, trainclasses)

y_predict2 = cl_LR.predict(testattributes)


clf = svm.SVC().fit(trainattributes, trainclasses)

y_predict3 = clf.predict(testattributes)


rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=0)
rf.fit(trainattributes, trainclasses)
y_predict4 = rf.predict(testattributes)


cm = np.array(confusion_matrix(testclasses, y_predict2))
noofposneg1 = y_predict2
df1 = pd.DataFrame({'0':noofposneg1})
qwe2 = df1['0'].value_counts()
ss1 = qwe2.loc[0]
ww1 = qwe2.loc[1]
xx1 = qwe2.loc[2]
yy1 = qwe2.loc[-1]
zz1 = qwe2.loc[-2]


pp1 = ss1/(ss1+ww1+xx1+yy1+zz1)
pp2 = ww1/(ss1+ww1+xx1+yy1+zz1)
pp3 = xx1/(ss1+ww1+xx1+yy1+zz1)
pp4 = yy1/(ss1+ww1+xx1+yy1+zz1)
pp5 = zz1/(ss1+ww1+xx1+yy1+zz1)



true_pos = np.diag(cm)
false_pos = np.sum(cm, axis=0) - true_pos
false_neg = np.sum(cm, axis=1) - true_pos

"""
precision = np.sum(true_pos / true_pos+false_pos)
recall = np.sum(true_pos / true_pos + false_neg)
"""
precision = np.sum(true_pos) / np.sum(cm, axis=0)
recall = np.sum(true_pos / np.sum(cm, axis=1))
accuracy = np.sum(true_pos) / l

emd = (np.absolute(pp1-p1))+(np.absolute(pp2-p2))+(np.absolute(pp3-p3))+(np.absolute(pp4-p4))+(np.absolute(pp5-p5))
print("Earth's mover distance",emd)








