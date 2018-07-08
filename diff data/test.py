import pickle
import pandas as pd

#dataset = pd.read_csv('train.tsv', delimiter = '\t', quoting = 3, header = None)
dataset = pd.read_csv('fake_or_real_news.csv')

import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#cv = CountVectorizer(max_features = 7500, stop_words = 'english', lowercase = True)
#X = cv.fit_transform(dataset[2][:]).toarray()


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words = 'english', lowercase = True, token_pattern = '[^a-zA-Z]') #max_features = 7500
X = cv.fit_transform(dataset.iloc[:,2]).toarray()

corp = []
string = ['U.S. Secretary of State John F. Kerry said Monday that he will stop in Paris later this week, amid criticism that no top American officials attended Sunday’s unity march against terrorism']
for i in range(1) :
    news = re.sub('[^a-zA-Z]',' ', string[i])
    news = news.lower()
    news = news.split()
    ps = PorterStemmer()
    news = [ps.stem(word) for word in news if not word in set(stopwords.words('english'))]
    news = ' '.join(news)
    corp.append(news)

y_s = cv.transform(corp).toarray()


# Opening the pickle file
# The rb stands for read binary
model_pkl = open("Random_forest_model_2.pkl", "rb")


# Reading the model
model = pickle.load(model_pkl)



# Calling the model# Calli 
#model


# Confirming the number of features
#model.n_features_


# The importance of each feature
#model.feature_importances_



# Testing the probability of a positive outcome of a new example# Testi 
result = model.predict(y_s)

def result_ans(result_a) :
    if result_a == 0 :
        return 'barely-true'
    elif result_a == 1 :
        return 'false'
    elif result_a == 2 :
        return 'half-true'
    elif result_a == 3 :
        return 'mostly-true'
    elif result_a == 4 :
        return 'false'
    elif result_a == 5 :
        return 'true'
    
