import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import SGDClassifier


dataset = pd.read_csv('fake_or_real_news.csv')


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words = 'english', lowercase = True, token_pattern = '[^a-zA-Z]') #max_features = 7500
X = cv.fit_transform(dataset.iloc[:,2]).toarray()
#y = dataset.iloc[:, 3].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(dataset.iloc[:,3])


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


clf = RandomForestClassifier(random_state=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)



import pickle
# Open the file to save as pkl file
# The wb stands for write and binary
model_pkl = open("Random_forest_model_2.pkl", "wb")

# Write to the file (dump the model)
# Open the file to save as pkl file
pickle.dump(clf, model_pkl)

# Close the pickle file
model_pkl.close()


