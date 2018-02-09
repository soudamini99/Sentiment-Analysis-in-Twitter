import pandas as pd
from sklearn import datasets, linear_model
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB 
from sklearn.metrics import roc_auc_score
import  csv, re, json
import preprocess_taskBD
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn import svm
from sklearn.metrics import f1_score
import numpy as np
    
df= pd.read_csv("data4.txt", sep='\t', names=['tweetid','topic','sentiment','tweet'])
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
l = len(testclasses)

cl_NB = MultinomialNB().fit(trainattributes, trainclasses)
y_predict1 = cl_NB.predict(testattributes)

rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)
rf.fit(trainattributes, trainclasses)
y_predict4 = rf.predict(testattributes)

cl_LR = linear_model.LogisticRegression(C=1e5)

cl_LR.fit(trainattributes, trainclasses)

y_predict3 = cl_LR.predict(testattributes)

clf = svm.SVC().fit(trainattributes, trainclasses)
y_predict2 = clf.predict(testattributes)



tn, fp, fn, tp = confusion_matrix(testclasses, y_predict1).ravel()
accuracy = (tp+tn)/(tp+tn+fn+fp);
precision = tp/(tp+fp);
recall = tp/(tp+fn);
fscore = 2*precision*recall/(precision+recall);

noofpos = (tp+tn)/(tp+tn+fp+fn)
noofneg = (fp+fn)/(tp+tn+fp+fn)
noofpos1 = ww/(ww+ss)
noofneg1 = ss/(ww+ss)
actual = np.array([noofpos1, noofneg1])
model1 = np.array([noofpos, noofneg])

epslon = 1/(2*l)
p_actual = (actual + epslon)/(1+(epslon*2))
p_model1 = (model1 + epslon)/(1+(epslon*2))

kl1 = (p_actual*np.log2(p_actual/p_model1)).sum()*l

abserror = (np.absolute(noofpos1-noofpos)/2)+(np.absolute(noofneg1-noofneg)/2)
error= ((np.absolute(p_model1-p_actual))/(p_actual*2)).sum()

print("Kullback-Leibler Divergence",kl1)



























