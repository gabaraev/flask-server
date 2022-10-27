#!/usr/bin/env python
# coding: utf-8

# In[29]:


import sys
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import BallTree
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline


sys.stdout.reconfigure(encoding="utf-8")


def softmax(x):
    proba = np.exp(-x)
    return proba / sum(proba)


class NeighborSampler(BaseEstimator):
    def __init__(self, k=1, temperature=1.0):
        self.k = k
        self.temperature = temperature

    def fit(self, X, y):
        self.tree = BallTree(X)
        self.y = np.array(y)

    def predict(self, X, random_state=None):
        distances, indices = self.tree.query(X, return_distance=True, k=self.k)
        result = []
        for distance, index in zip(distances, indices):
            result.append(
                np.random.choice(index, p=softmax(distance * self.temperature))
            )
        return self.y[result]

base_dir = os.getcwd()
database = pd.read_csv(
    "database.csv", sep=";", on_bad_lines="skip"
)  # в первом аргументе необходимо указать путь до файла с данными


vectorizer = CountVectorizer()
vectorizer.fit(database.context_0)
matrix = vectorizer.transform(database.context_0)

svd = TruncatedSVD(n_components=280)
svd.fit(matrix)
new_matrix = svd.transform(matrix)


ns = NeighborSampler()
ns.fit(new_matrix, database.resp)



pipe = make_pipeline(vectorizer, svd, ns)

def getAnswer(request):
    answer = pipe.predict([request])
    ns.fit(new_matrix, database.topic)
    topic = pipe.predict([request])
    return {"type": topic[0], "content": answer[0]}

# print(
#     answer_and_topic
# )  # выведет в формате {'general': 'Привет! Чем я могу тебе помочь?'}
# # sys.stdout.flush()
