import pandas as pd
import os
import pyprind
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

pbar = pyprind.ProgBar(50000)
labels = {'pos':1, 'neg':0}
df = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path ='./aclImdb/%s/%s' % (s, l)
        for file in os.listdir(path):
            with open(os.path.join(path, file), 'r') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]], ignore_index=True)
            pbar.update()
df.columns = ['review', 'sentiment']

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('./movie_data.csv', index=False)
df = pd.read_csv('./movie_data.csv')
print df.head(3)

count = CountVectorizer()
docs = np.array(['The sun is shining', 'The weather is sweet', 'The sun is shining and the weather is sweet'])
bag = count.fit_transform(docs)
print bag
print(count.vocabulary_)
print(bag.toarray())