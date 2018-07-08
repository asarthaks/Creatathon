max_features = 7500



import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(6336) :
    news = re.sub('[^a-zA-Z]',' ', dataset.iloc[i][2])
    news = news.lower()
    news = news.split()
    ps = PorterStemmer()
    news = [ps.stem(word) for word in news if not word in set(stopwords.words('english'))]
    news = ' '.join(news)
    corpus.append(news)
    
    
clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
clf4 = SGDClassifier(loss="hinge", penalty="l2")

eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3), ('sgdc', clf4)],  voting='hard')

for clf, label in zip([clf1, clf2, clf3, clf4, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Stochastic gradientC', 'Ensemble']):
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    
    
    
    
    
    
    

import numpy as np
d = np.loadtxt(path, delimiter="\t")
print d[0,2] # 248


writer = pd.ExcelWriter("data_real.xlsx",
                        engine='xlsxwriter')
 
# Convert the dataframe to an XlsxWriter Excel object.
dataset.to_excel(writer, sheet_name='Sheet1')
 
# Get the xlsxwriter workbook and worksheet objects in order to set the column
# widths, to make the dates clearer.
workbook  = writer.book
worksheet = writer.sheets['Sheet1']
 
worksheet.set_column('B:C', 20)
 
# Close the Pandas Excel writer and output the Excel file.
writer.save()

import xlrd
import csv

# open the output csv
with open('my.csv', 'wb') as myCsvfile:
    # define a writer
    wr = csv.writer(myCsvfile, delimiter="\t")

    # open the xlsx file 
    myfile = xlrd.open_workbook('data_real.xlsx')
    # get a sheet
    mysheet = myfile.sheet_by_index(0)

    # write the rows
    for rownum in range(mysheet.nrows):
        wr.writerow(mysheet.row_values(rownum))
        
        
        
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

