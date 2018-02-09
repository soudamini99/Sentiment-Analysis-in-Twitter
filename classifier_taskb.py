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
import preprocess_taskBD
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn import svm
from sklearn.metrics import f1_score
    
df= pd.read_csv("data4.txt", sep='\t', names=['tweetid','topic','sentiment','tweet'])


stopset = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(use_idf=True,lowercase=True, strip_accents='ascii', stop_words=stopset)

y= df.sentiment

d_y = y



x= vectorizer.fit_transform(df.tweet)

print (y.shape)
print (x.shape)

trainattributes, testattributes, trainclasses, testclasses = train_test_split (x, y, random_state=4000 )



cl_LR = linear_model.LogisticRegression(C=1e5)

cl_LR.fit(trainattributes, trainclasses)

y_predict3 = cl_LR.predict(testattributes)

cl_NB = MultinomialNB().fit(trainattributes, trainclasses)
y_predict1 = cl_NB.predict(testattributes)

clf = svm.SVC().fit(trainattributes, trainclasses)
y_predict2 = clf.predict(testattributes)


cm = np.array(confusion_matrix(testclasses, y_predict3))
cm1 = cm[0][0]
cm2 = cm[0][1]
cm3 = cm[1][0]
cm4 = cm[1][1]

accuracy = (cm1+cm4)/(cm1+cm2+cm3+cm4);
pos_precision = cm1/(cm1+cm3)
neg_precision = cm4/(cm4+cm2)
pos_recall = cm1/(cm1+cm2)
neg_recall = cm4/(cm4+cm3)

fpos = (2*pos_precision*pos_recall)/(pos_precision+pos_recall)
fneg = (2*neg_precision*neg_recall)/(neg_precision+neg_recall)

fscore = (fpos+fneg)/2
rho = (pos_recall+neg_recall)/2

print("\n")
print("Accuracy",accuracy)
print("F Score",fscore)
print("Macroaveraged recall",rho)

















