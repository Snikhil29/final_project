import sys
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import *
import re
import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier,LinearRegression,BayesianRidge,passive_aggressive,ARDRegression
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,VotingClassifier,AdaBoostClassifier,BaggingClassifier,ExtraTreesClassifier
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from nltk.corpus import stopwords
from pandas import DataFrame
from sklearn.svm import SVC, LinearSVC, NuSVC
import matplotlib.pyplot as plt


data_dir_train = 'C:\\Users\\NIKHIL\\Desktop\\7th sem minr\\db\\reviews\\train'
data_dir_test = 'C:\\Users\\NIKHIL\\Desktop\\7th sem minr\\db\\reviews\\test'
classes = ['pos', 'neg']


# converting reviews to words
def Convert_Rev_To_Words(raw_review):
    letters_only = re.sub("[^a-zA-Z]", " ", raw_review)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    return (" ".join(meaningful_words))


# Read the data
train_data = []
train_labels = []
train_fnames = []
test_data = []
test_labels = []
test_fnames = []

# Loading train data
print 'Loading train data...'
for curr_class in classes:
    dirname = os.path.join(data_dir_train, curr_class)
    for fname in os.listdir(dirname):
        with open(os.path.join(dirname, fname), 'r') as f:
            content = f.read()
            id, ext = os.path.splitext(fname)
            processedContent = Convert_Rev_To_Words(content)
            train_data.append(processedContent)
            train_fnames.append(id)
            if (curr_class == 'pos'):
                tag = 1
            if (curr_class == 'neg'):
                tag = 0
            train_labels.append(tag)
print 'Loading train data Successfull'

print 'Loading test data...'

for curr_class in classes:
    dirname = os.path.join(data_dir_test, curr_class)
    for fname in os.listdir(dirname):
        with open(os.path.join(dirname, fname), 'r') as f:

            if (curr_class == 'pos'):
                tag = 1
                content = f.read()
                id, ext = os.path.splitext(fname)
                processedContent = Convert_Rev_To_Words(content)
                test_data.append(processedContent)
                test_fnames.append(id)
                test_labels.append(tag)
            if (curr_class == 'neg'):
                tag = 0
                content = f.read()
                id, ext = os.path.splitext(fname)
                processedContent = Convert_Rev_To_Words(content)
                test_data.append(processedContent)
                test_fnames.append(id)
                test_labels.append(tag)
# u.write("\"%s\"\t%s\t\"%s\"\n" % (id,tag,content))
#           w.write("\"%s\"\t%s\t\"%s\"\n" % (id,tag,processedContent))
print 'Loading test data Successfull'

num_reviews = len(train_data)
meaning_train_reviews = []
print num_reviews
for i in range(0, num_reviews):
    meaning_train_reviews.append(Convert_Rev_To_Words(train_data[i]))
print ("cleaning train complete")

meaning_test_reviews = []
num_reviews_test = len(test_data)
print num_reviews_test
for i in range(0, num_reviews_test):
    meaning_test_reviews.append(Convert_Rev_To_Words(test_data[i]))

v = CountVectorizer(analyzer = "word",
                    ngram_range=(1, 3),
                    token_pattern=r'\b\w+\b',
                    tokenizer = None,
                    preprocessor = None,
                    stop_words = None,
                    max_features =1000)

train_data_features1 = v.fit_transform(meaning_train_reviews)
test_data_features1=v.transform(meaning_test_reviews)

results=[]
"""
models=[]
models.append(('RF',RandomForestClassifier()))
models.append(('ADA',AdaBoostClassifier()))
models.append(('GB',GradientBoostingClassifier()))
models.append(('BAG',BaggingClassifier()))
models.append(('RF',ExtraTreesClassifier()))

for name,model in models:
"""

print "Random_Forest_Classifier"
forest = RandomForestClassifier(n_estimators=100)
forest_classifier = forest.fit(train_data_features1, train_labels)
forest_result = forest_classifier.predict(test_data_features1)
forest_output = pd.DataFrame( data={"id":test_fnames, "sentiment":forest_result,"given":test_labels })
forest_output.to_csv( "output1\\resultF3.csv", index=False, quoting=3 )

train_data_features2 = v.fit_transform(meaning_train_reviews).toarray()

test_data_features2=v.transform(meaning_test_reviews).toarray()
print "Gradient_Boosting_Classifier"
GradientBoosting = GradientBoostingClassifier(n_estimators=100)
GradientBoosting_classifier = GradientBoosting.fit(train_data_features1, train_labels)
GradientBoosting_result = GradientBoosting_classifier.predict(test_data_features2)
GradientBoosting_output = pd.DataFrame( data={"id":test_fnames, "sentiment":GradientBoosting_result,"given":test_labels })
GradientBoosting_output.to_csv( "output1\\resultGB3.csv", index=False, quoting=3 )

print "AdaBoost_Classifier"
AdaBoost = AdaBoostClassifier(n_estimators=100)
AdaBoost_Classifier = AdaBoost.fit(train_data_features1, train_labels)
AdaBoost_result = AdaBoost_Classifier.predict(test_data_features2)
AdaBoost_output = pd.DataFrame( data={"id":test_fnames, "sentiment":AdaBoost_result,"given":test_labels })
AdaBoost_output.to_csv( "output1\\resultAB3.csv", index=False, quoting=3 )


print "Bagging_Classifier"
Bagging = BaggingClassifier(n_estimators=100)
Bagging_Classifier = Bagging.fit(train_data_features1, train_labels)
Bagging_result = Bagging_Classifier.predict(test_data_features2)
Bagging_output = pd.DataFrame( data={"id":test_fnames, "sentiment":Bagging_result,"given":test_labels })
Bagging_output.to_csv( "output1\\resultBAG3.csv", index=False, quoting=3 )


print "ExtraTrees_Classifier"
ExtraTrees = ExtraTreesClassifier(n_estimators=100)
ExtraTrees_Classifier = ExtraTrees.fit(train_data_features1, train_labels)
ExtraTrees_result = ExtraTrees_Classifier.predict(test_data_features2)
ExtraTrees_output = pd.DataFrame( data={"id":test_fnames, "sentiment":ExtraTrees_result,"given":test_labels })
ExtraTrees_output.to_csv( "output1\\resultET3.csv", index=False, quoting=3 )
