# Sentiment-Analysis-in-Twitter
This task is a part of SemEval 2016 - Sentiment Analysis in Twitter.
Sentiment is a way to measure the tone of conversation whether the person is annoyed, happy and angry towards anything. Sentiment Analysis is a process of analysing these emotional tones that helps to gain the understanding of attitudes, emotions and an overview of public opinions which is extremely used in social media. Our task is to classify a tweet based on the sentiment conveyed by it and the major enhancements to Semeval 2015 task 10 are that we replaced ordinal classification with quantification and moved from two point scale to five point scale. The major enhancements from SemEval 2015  Task 10 to 2016 Task 4 are we replaced ordinal classification with quantification and moved from two point scale to five point scale.
#Feature Extraction
The text is pre-processed by normalizing the URL’s and mentions of the users and the hash-tags to constants like http://someurl and @someuser. 
Here, we removed the elongated text, converted the text to lowercase after which Parts Of Speech tagging is performed. (e.g., Happpyyyyyy). The features included here are all-caps tokens, the number of tokens for each POS tags, number of hash-tags, number of negated contexts, number of sequence of exclamation or question marks, elongated words. 
A key addition to the above features mentioned is the vector representation of the data, i.e., TF-IDF. 
The data is divided into training and test sets.
#Subtask A
For this task, a Support Vector Machine with a Linear Kernal and a Multinomial Naïve Bayes classifier is trained on the dataset with the above mentioned n unigram features. 
The class value of the tuple is used as the correct label of the feature vector. 
Similarly, for each test sentence, a feature vector is filled and the trained SVM is used to predict the probabilities of assigning each possible category to sentence as Positive, Negative or Neutral. 
The evaluation metrics F-scores of positive and negative classes, macro averaged recall and accuracy are calculated from the confusion matrix. 
#Subtask B 
For this task, a Support Vector Machine with a Linear Kernal and a Multinomial Naïve Bayes classifier is trained on the dataset with the above mentioned n unigram features.
The class value of the tuple is used as the correct label of the feature vector for each tweet. 
Similarly, for each test sentence, a feature vector is filled and the trained SVM is used to predict the probabilities of assigning each possible category to sentence as Positive or Negative sentiment towards the topic.
The evaluation metrics are same as subtask A.
#Subtask C
Here, we have five category values on an ordinal scale and it is an ordinal classification problem (also known as ordinal regression.
The main difference between subtask A and subtask C is all mistakes weigh equally; e.g; classifying an item as very negative that should be actually classified as very positive is a serious mistake than classifying an item as very negative that should be classified as negative.
We implemented random decision forests and ordinal logistic regression with the selected n unigram features.
The class value of the tuple is used as the correct label of the feature vector for each tweet. 
Similarly, for each test sentence, a feature vector is filled and the trained model predicts the probabilities of assigning each possible category to sentence as HighlyPositive, Positive, Neutral, Negative and HighlyPegative sentiment towards the topic.
Macro averaged mean absolute error is used as a standard evaluation metric.
#Subtask D
Here, we identified the distribution of positive and negative tweets towards the topic on the datasets. 
We trained the dataset with Linear Logistic regression model and Multinomial Naïve Bayes with the selected n unigram features.
We implemented this model on the test set and calculated the prediction of distribution of the tweets on positive and negative classes.
Kullback Leibler Divergence measures the  dissimilarity over the predicted distribution and actual distribution of positive and negative classes.
#Subtask D
Here, we identified the distribution of positive and negative tweets towards the topic on the datasets. 
We trained the dataset with Linear Logistic regression model and Multinomial Naïve Bayes with the selected n unigram features.
We implemented this model on the test set and calculated the prediction of distribution of the tweets on positive and negative classes. 
Kullback Leibler Divergence measures the  dissimilarity over the predicted distribution and actual distribution of positive and negative classes.
