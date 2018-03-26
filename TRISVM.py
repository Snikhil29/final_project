import os
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from pandas import DataFrame
from sklearn.svm import SVC, LinearSVC, NuSVC

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

v = CountVectorizer(analyzer = "word", \
                    ngram_range=(1, 3),
                    tokenizer = None,    \
                    preprocessor = None, \
                    stop_words = None,   \
                    max_features = 150)

train_data_features1 = v.fit_transform(meaning_train_reviews)
test_data_features1=v.transform(meaning_test_reviews)


train_data_features2 = v.fit_transform(meaning_train_reviews).toarray()
test_data_features2=v.transform(meaning_test_reviews).toarray()


print("--------------------------------------SVM MODELS-----------------------------------------------------")
print("--------------------------------------SVM MODELS-----------------------------------------------------")
print("--------------------------------------SVM MODELS-----------------------------------------------------")

print("LinearSVC-----------------------------------------------------")

LinearSVC_classifier = LinearSVC().fit(train_data_features1, train_labels)
LinearSVC_result = LinearSVC_classifier.predict(test_data_features1)
LinearSVC_output = pd.DataFrame(data={"id": test_fnames, "sentiment": LinearSVC_result, "given": test_labels})
LinearSVC_output.to_csv("output1\\resultLSVC3.csv", index=False, quoting=3)


print("SVC-----------------------------------------------------")

SVC_classifier = SVC().fit(train_data_features1, train_labels)
SVC_result = SVC_classifier.predict(test_data_features1)
SVC_output = pd.DataFrame(data={"id": test_fnames, "sentiment": SVC_result, "given": test_labels})
SVC_output.to_csv("output1\\resultSVC3.csv", index=False, quoting=3)

print("NuSVC-----------------------------------------------------")

NuSVC_classifier = NuSVC().fit(train_data_features1, train_labels)
NuSVC_result = NuSVC_classifier.predict(test_data_features1)
NuSVC_output = pd.DataFrame(data={"id": test_fnames, "sentiment": NuSVC_result, "given": test_labels})
NuSVC_output.to_csv("output1\\resultNuSVC3.csv", index=False, quoting=3)


print ("done ")

