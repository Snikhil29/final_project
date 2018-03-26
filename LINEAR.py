import os
import re
from sklearn.linear_model import LogisticRegression, SGDClassifier,BayesianRidge
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

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

v = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 250)

train_data_features1 = v.fit_transform(meaning_train_reviews)
test_data_features1=v.transform(meaning_test_reviews)


train_data_features2 = v.fit_transform(meaning_train_reviews).toarray()
test_data_features2=v.transform(meaning_test_reviews).toarray()


print("-------------------------------linear MODELS-----------------------------------------------------")
print("-------------------------------linear MODELS-----------------------------------------------------")
print("-------------------------------linear MODELS-----------------------------------------------------")

print("SGDClassifier-----------------------------------------------------")
SGDClassifier_classifier = SGDClassifier().fit(train_data_features1, train_labels)
SGDClassifier_result = SGDClassifier_classifier.predict(test_data_features1)
SGDClassifier_output = pd.DataFrame( data={"id":test_fnames, "sentiment":SGDClassifier_result,"given":test_labels })
SGDClassifier_output.to_csv( "output\\resultSGDC1.csv", index=False, quoting=3 )

print "LogisticRegression_Classifier"
LogisticRegression_classifier = LogisticRegression().fit(train_data_features1, train_labels)
LogisticRegression_result = LogisticRegression_classifier.predict(test_data_features1)
LogisticRegression_output = pd.DataFrame( data={"id":test_fnames, "sentiment":LogisticRegression_result,"given":test_labels })
LogisticRegression_output.to_csv( "output\\resultLOR1.csv", index=False, quoting=3 )


print "BayesianRidge_Classifier"
BayesianRidge_classifier = BayesianRidge().fit(train_data_features2, train_labels)
BayesianRidge_result = BayesianRidge_classifier.predict(test_data_features2)
BayesianRidge_output = pd.DataFrame( data={"id":test_fnames, "sentiment":BayesianRidge_result,"given":test_labels })
BayesianRidge_output.to_csv( "output\\resultBR1.csv", index=False, quoting=3 )
print "LinearModel done"