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




cl_NB = MultinomialNB().fit(trainattributes, trainclasses)

y_predict1 = cl_NB.predict(testattributes)


cl_LR = linear_model.LogisticRegression(C=1e5)

cl_LR.fit(trainattributes, trainclasses)

y_predict2 = cl_LR.predict(testattributes)
#metrics.confusion_matrix(testclasses, y_predict)

clf = svm.SVC().fit(trainattributes, trainclasses)

y_predict3 = clf.predict(testattributes)
rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)
rf.fit(trainattributes, trainclasses)
y_predict4 = rf.predict(testattributes)

#[l,m] = testattributes.shape
l = len(testclasses)
cm = np.array(confusion_matrix(testclasses, y_predict1))
true_pos = np.diag(cm)
false_pos = np.sum(cm, axis=0) - true_pos
false_neg = np.sum(cm, axis=1) - true_pos


recall = np.sum(true_pos / np.sum(cm, axis=1))
accuracy = np.sum(true_pos) / l


def mean_absolute_error(x,y):
    mae = ((np.absolute(y-x).sum())/l)
    return mae

def compute_score(x, y):
        """It computes the Macro-averaged Mean Absolute Error (MAE), normalize
        it wrt the highest possible error and convert it into a percentage
        score. The score ranges between 0 and 1:
         - 000: lowest score;
         - 100: highest score;
        """
        mae_per_score = []
        stats_per_score = dict()
        for score in set(x):
            result = 0           
            x_n, y_n = zip(*[(a, b) for a, b in zip(x, y) if a == score])
            
            if score in (-2, 2):
                normalisation_factor = 5
            else:
                normalisation_factor = 3
                        try:
                result = 100 * (1 - (mean_absolute_error(testclasses, y_predict1) / normalisation_factor))
        
            except ValueError:
                result = 0

            mae_per_score.append(result)
        score = sum(mae_per_score) / len(set(x))
    
        return score/100


mae =  mean_absolute_error(testclasses, y_predict1)   
macromae = compute_score(testclasses,y_predict1)

print("Standard mean absolute error:", mae)
print("Macroaveraged mean absolue error:", macromae)




















