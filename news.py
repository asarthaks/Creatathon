import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('train.tsv', delimiter = '\t', quoting = 3, header = None)

import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(10269) :
    news = re.sub('[^a-zA-Z]',' ', dataset[2][i])
    news = news.lower()
    news = news.split()
    ps = PorterStemmer()
    news = [ps.stem(word) for word in news if not word in set(stopwords.words('english'))]
    news = ' '.join(news)
    corpus.append(news)

    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 7500)
X = cv.fit_transform(corpus).toarray()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(dataset.iloc[:,1])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
#from sklearn.naive_bayes import GaussianNB
#classifier = GaussianNB()

#from sklearn.ensemble import RandomForestClassifier
#classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)

