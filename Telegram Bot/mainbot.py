# -*- coding: UTF8 -*-
import requests
import datetime
import pickle
import pandas as pd

dataset = pd.read_csv('fake_or_real_news.csv')

import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words = 'english', lowercase = True, token_pattern = '[^a-zA-Z]') #max_features = 7500
X = cv.fit_transform(dataset.iloc[:,2]).toarray()

# Opening the pickle file
# The rb stands for read binary
model_pkl = open("Random_forest_model_2.pkl", "rb")


# Reading the model
model = pickle.load(model_pkl)


def result_ans(result_a) :
    if result_a == 0 :
        return 'false'
    elif result_a == 1 :
        return 'true'




class BotHandler:
    def __init__(self, token):
            self.token = token
            self.api_url = "https://api.telegram.org/bot{}/".format(token)

    #url = "https://api.telegram.org/bot<token>/"

    def get_updates(self, offset=0, timeout=30):
        method = 'getUpdates'
        params = {'timeout': timeout, 'offset': offset}
        resp = requests.get(self.api_url + method, params)
        result_json = resp.json()['result']
        return result_json

    def send_message(self, chat_id, text):
        params = {'chat_id': chat_id, 'text': text, 'parse_mode': 'HTML'}
        method = 'sendMessage'
        resp = requests.post(self.api_url + method, params)
        return resp

    def get_first_update(self):
        get_result = self.get_updates()

        if len(get_result) > 0:
            last_update = get_result[0]
        else:
            last_update = None

        return last_update


token = '564061413:AAHyr8MXiXC7XgQYcKcT-2WB_CWiLZSuK-k' #Token of your bot
magnito_bot = BotHandler(token) #Your bot's name



def main():
    new_offset = 0
    print('hi, now launching...')

    while True:
        all_updates=magnito_bot.get_updates(new_offset)

        if len(all_updates) > 0:
            for current_update in all_updates:
                print(current_update)
                first_update_id = current_update['update_id']
                if 'text' not in current_update['message']:
                    first_chat_text='New member'
                else:
                    first_chat_text = current_update['message']['text']
                first_chat_id = current_update['message']['chat']['id']
                if 'first_name' in current_update['message']:
                    first_chat_name = current_update['message']['chat']['first_name']
                elif 'new_chat_member' in current_update['message']:
                    first_chat_name = current_update['message']['new_chat_member']['username']
                elif 'from' in current_update['message']:
                    first_chat_name = current_update['message']['from']['first_name']
                else:
                    first_chat_name = "unknown"

                if first_chat_text == 'Hi':
                    magnito_bot.send_message(first_chat_id, 'Hello ' + first_chat_name)
                    new_offset = first_update_id + 1
            	    
                elif first_chat_text[:3] == '/mr':
                    magnito_bot.send_message(first_chat_id, first_chat_text[4:])
                    new_offset = first_update_id + 1
                elif first_chat_text[:3] == '/fn':
                    corp = []
                    string = [first_chat_text[4:]]
                    for i in range(1) :
                        news = re.sub('[^a-zA-Z]',' ', string[i])
                        news = news.lower()
                        news = news.split()
                        ps = PorterStemmer()
                        news = [ps.stem(word) for word in news if not word in set(stopwords.words('english'))]
                        news = ' '.join(news)
                        corp.append(news)
                    
                    y_s = cv.transform(corp).toarray()
                    result = model.predict(y_s)
                    res = result_ans(result)
                    magnito_bot.send_message(first_chat_id, res)
                    new_offset = first_update_id + 1
                else:
                    magnito_bot.send_message(first_chat_id, 'Welcome, use the command /fn at starting of a news to ' +
                                             'check if the news is fake or not')
                    new_offset = first_update_id + 1


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        exit()

