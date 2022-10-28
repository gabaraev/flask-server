#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import BallTree
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline

nltk.download("stopwords")
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


def text_prepare(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    text = re.sub(r'[^\w\s]','', text.lower())
    words = nltk.word_tokenize(text)
    stop_words = stopwords.words("russian")
    stop_words.remove('кто')
    stop_words.remove('ты')
    stop_words.remove('как')
    stop_words.remove('что')
    
    
    lemmatized_words = []
    
    for w in words:
        lemmatized_word = wordnet_lemmatizer.lemmatize(w)
        if not lemmatized_word in stop_words:
            lemmatized_words.append(wordnet_lemmatizer.lemmatize(w))
    
    
    lemmatized_text = " ".join(lemmatized_words)
    return lemmatized_text
    
    

def key_word_search(req):
    
    
    key_words = {'dorm': ['общаг', 'общежити'], 
                     'club': ['студклуб', 'клуб', 'творчеств'],
                     'docs': ['документ', 'справк', 'док'],
                     'grant': ['стипенд', 'стипух', 'деньги'],
                     'military':['военн', 'военка', 'вуц'],
                     'open_doors': ['двер', 'открыт'],
                     'specialization': ['специальност', 'направлен', 'факультет'],
                     'sport': ['спорт', 'секци'],
                     }
                     
    responces = {'dorm': 'Всего у ЮФУ 19 общежитий. Из них 12 расположены в Ростове-на-Дону и 7 в Таганроге. Более подробную информацию о количестве общежитий, типе общежитий и количестве мест можно узнать вот тут: https://sfedu.ru/www/stat_pages22.show?p=STD/N13054/P', 
                     'club': 'Студенческий клуб - это про творчество! Тут есть много разных направлений. Подробнее можно почитать в статье: https://vk.com/@studclub_tgn-chto-takoe-sk',
                     'docs': 'Найти перечень документов для поступления можно по этой ссылке: https://sfedu.ru/www/stat_pages22.show?p=ABT/main/M',
                     'grant': 'Подробнее о видах стипендий и их размере можно узнать вот тут: https://sfedu.ru/www/stat_pages22.show?p=STI/main/M',
                     'military':'В ЮФУ есть Военный учебный центр. Подробнее можно почитать по ссылке: https://sfedu.ru/www/stat_pages22.show?p=ABT/N8227/P',
                     'open_doors': 'Всю информацию о днях открытых дверей можно найти по ссылке: https://sfedu.ru/www/stat_pages22.show?p=ABT/N8219/P',
                     'specialization': '''Вы можете узнать все в этой таблице: https://sfedu.ru/www/stat_pages22.show?p=ABT/N8206 .
Достаточно просто нажать на интересующую вас специальность. Вас отправят на страницу этой специальности, где можно подробно ознакомиться с информацией о ней и даже задать вопросы :))''',
                     'sport': 'В рамках занятий по физической культуре есть возможность заниматься совершенно разными видами спорта( от плаванья до борбы). Актуальную запись можно найти в личном кабинете студента в разделе "Запись".\nПомимо этого есть платные секции, о них можно узнать у куратора.\nВ ЮФУ постоянно проводятся различные спортивные соревнования!',
                     }
        
    for key, value in key_words.items():
                for item in value:
                    if item in req:
                         return key, responces[key]
                        
    return 'general', 'Извините, но похоже я вас не понимаю. Попробуйте перефразировать вопрос или же обратиться в приемную комиссию'



class NeighborSampler(BaseEstimator):
    def __init__(self, k = 1, temperature = 1.0):
        self.k = k
        self.temperature = temperature
    
    def fit(self, X, y):
        self.tree = BallTree(X)
        self.y = np.array(y)
    
    def predict(self, X, random_state = None):
        
       
        distances, indices = self.tree.query(X, k = self.k, return_distance = True)
        result = []
        for distance, index in zip(distances, indices):
            if distance * 0.7 < 0.7:
                 result.append(np.random.choice(index)) 
                            
        
        return self.y[result]
        
        
def predict_answer(text):
    request = text # сообщение пользователя
    
    if request.isascii():
        topic, answer = 'general', 'London is the capital of Great Britain! Это единственное, что я могу сказать вам на английском. Пожалуйста, задайте вопрос на русском языке, так я вероятнее смогу помочь вам :)'
        answer_and_topic = {'type': topic, 'content': answer}

    else:
        ns.fit(new_matrix, database.resp)
        answer = pipe.predict([request])
        if answer.size > 0:
            ns.fit(new_matrix, database.topic)                    
            topic = pipe.predict([request])                     
            answer_and_topic = {'type': topic[0], 'content': answer[0]}
        else:
            topic, answer = key_word_search(request)
            answer_and_topic = {'type': topic, 'content': answer}
    
    
    return answer_and_topic
        
    
    
    
    
database = pd.read_csv('database.csv', sep = ';', on_bad_lines ='skip') # в первом аргументе необходимо указать путь до файла с данными

database['text_lemmatized'] = database.context_0.apply(text_prepare)


vectorizer = CountVectorizer()
vectorizer.fit(database.text_lemmatized)
matrix = vectorizer.transform(database.text_lemmatized)


svd = TruncatedSVD(n_components = 300)
svd.fit(matrix)
new_matrix = svd.transform(matrix)

ns = NeighborSampler()

pipe = make_pipeline(vectorizer, svd, ns)


# print(predict_answer('Расскажи про общежития')) # выведется результат функции, которая предсказывает ответ






