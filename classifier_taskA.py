import pandas as pd
import numpy as np
from sklearn import datasets, linear_model
import nltk
from sklearn.model_selection import cross_val_score
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB 
from sklearn.metrics import roc_auc_score
import  csv, re, json
import preprocess_taskA
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn import svm
from sklearn.metrics import f1_score
    
df= pd.read_csv("data14.txt", sep='\t', names=['tweetid','sentiment','tweet'])


stopset = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(use_idf=True,lowercase=True, strip_accents='ascii', stop_words=stopset)

y= df.sentiment

d_y = y



x= vectorizer.fit_transform(df.tweet)

print (y.shape)
print (x.shape)


trainattributes, testattributes, trainclasses, testclasses = train_test_split (x, y, random_state=0)

cl_LR = linear_model.LogisticRegression(C=1e5)

cl_LR.fit(trainattributes, trainclasses)

y_predict3 = cl_LR.predict(testattributes)







cl_NB = MultinomialNB().fit(trainattributes, trainclasses)

y_predict1 = cl_NB.predict(testattributes)

clf = svm.SVC().fit(trainattributes, trainclasses)

y_predict2 = clf.predict(testattributes)


l = len(testclasses)
cm = np.array(confusion_matrix(testclasses, y_predict3))


cm1 = cm[0][0]
cm2 = cm[0][1]
cm3 = cm[0][2]
cm4 = cm[1][0]
cm5 = cm[1][1]
cm6 = cm[1][2]
cm7 = cm[2][0]
cm8 = cm[2][1]
cm9 = cm[2][2]


pos_recall = cm1/(cm1+cm2+cm3)
pos_precision = cm1/(cm1+cm4+cm7)
neg_recall = cm5/(cm2+cm5+cm8)
neg_precision = cm5/(cm4+cm5+cm6)

true_pos = np.diag(cm)

fpos = (2*pos_precision*pos_recall)/(pos_precision+pos_recall)
fneg = (2*neg_precision*neg_recall)/(neg_precision+neg_recall)

Fpn = (fpos+fneg)/2
rhopos = (pos_recall+neg_recall)/2

accuracy = np.sum(true_pos) / l
print("Accuracy:",accuracy)
print("F Score:",Fpn)
print("Macroaveraged recall:",rhopos)



