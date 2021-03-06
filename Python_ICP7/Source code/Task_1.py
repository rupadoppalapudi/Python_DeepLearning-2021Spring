# Change the classifier in the given source code to
# a.  SVM and see how accuracy changes
# b. Set the tfidf vectorizer parameter to use bigram and see how the accuracy changes TfidfVectorizer(ngram_range=(1,2))
# c. Set tfidf vectorizer argument to use stop_words='english' and see how accuracy changes

# importing libraries
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

# preparing training data set
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

# list of target names
print("\nList :");
print(list(twenty_train.target_names))

cts = ['alt.atheism', 'rec.sport.baseball', 'sci.electronics', 'talk.religion.misc']
# preparing training data set for a few category of target names
twenty_train_cat = fetch_20newsgroups(subset='train', categories=cts, shuffle=True)

# applying TfidfVectorizer
tfidf_Vect = TfidfVectorizer()
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train_cat.data)

# training data with Multinomial Naive Bayes
clf = MultinomialNB()
clf.fit(X_train_tfidf, twenty_train_cat.target)

# preparing test data set for a few category of target names
twenty_test_cat = fetch_20newsgroups(subset='test', categories=cts, shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test_cat.data)

# prediction of the test data set
NB_predicted = clf.predict(X_test_tfidf)

# calculating accuracy score for the Multinomial Naive Bayes model
NB_score = metrics.accuracy_score(twenty_test_cat.target, NB_predicted)
print("\nAccuracy score using MultinomialNB model: ", NB_score)

####### SVM model
svc = SVC(kernel='linear', random_state=1)
svc.fit(X_train_tfidf, twenty_train_cat.target)

# prediction of the test data set with SVC
svc_predicted = svc.predict(X_test_tfidf)

# calculating Accuracy score using SVC model
svc_score = metrics.accuracy_score(twenty_test_cat.target, svc_predicted)
print("Accuracy score using SVM: ", svc_score)

########  Set the tfidf vectorizer parameter to use bigram

# Setting TfidfVectorizer(ngram_range=(1,2))
tfidf_Vect_bigram = TfidfVectorizer(ngram_range=(1, 2))

# preparing training data set
X_train_tfidf_bigram = tfidf_Vect_bigram.fit_transform(twenty_train_cat.data)
X_test_tfidf_bigram = tfidf_Vect_bigram.transform(twenty_test_cat.data)
svc.fit(X_train_tfidf_bigram, twenty_train_cat.target)

# prediction using SVC
svc_predicted_bigram = svc.predict(X_test_tfidf_bigram)

# calculating Accuracy score
svc_score_bigram = metrics.accuracy_score(twenty_test_cat.target, svc_predicted_bigram)
print("Accuracy score using Bigram TFIDF: ", svc_score_bigram)

###### Set tfidf vectorizer argument to use stop_words='english'

# Setting TfidfVectorizer(ngram_range=(1,2)) & Stopword=english
tfidf_Vect_stopword = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')

# preparing training data set
X_train_tfidf_stopwatch = tfidf_Vect_stopword.fit_transform(twenty_train_cat.data)
X_test_tfidf_stopwatch = tfidf_Vect_stopword.transform(twenty_test_cat.data)
svc.fit(X_train_tfidf_stopwatch, twenty_train_cat.target)

# prediction using SVC
svc_predicted_stopword = svc.predict(X_test_tfidf_stopwatch)

# calculating Accuracy score
svc_score_stopword = metrics.accuracy_score(twenty_test_cat.target, svc_predicted_stopword)
print("Accuracy score by using stopword: ", svc_score_stopword)