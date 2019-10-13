
print("Importing dependencies...")
import numpy as np
import re
import pandas as pd
import json
import gzip
import sqlite3
#from sqlitedict import SqliteDict

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
#from nltk import punkt
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

#nltk.download()
#from nltk import stopwords
nltk.download('stopwords')
nltk.download('punkt')

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

print("Loading data...")
df = pd.read_csv ('/Users/VishalDubey/Desktop/HackNC/househack/learning/data.csv')
pd.set_option('display.max_colwidth', -1)

#set up variables
num_paragraphs, num_pics, text, num_words, num_chars, num_tags, abv75p, abv50p, abv25p, bottom25p = [],[],[],[],[],[],[],[],[],[]

#used for ML
df1 = df[df.columns[-23:]]

#used to calculate what percentile by hits an article is in
p75 = np.percentile(df['Hits'], 75)
p50 = np.percentile(df['Hits'], 50)
p25 = np.percentile(df['Hits'], 25)

for row in df.itertuples():
    i = 0
    for x in row[12:]:
        if x:
            i += 1
    num_tags.append(i)
    
    num_paragraphs.append(len(re.findall(r"<br><br>", row.Body)))
    num_pics.append(len(re.findall(r"href=", row.Body)))
    text.append(re.sub(r'<[^<>]+>', '', row.Body))
    num_words.append(len(text[-1].split()))
    num_chars.append(len(text[-1]))
    
    bottom25p.append(int(row.Hits <= p25))
    abv25p.append(int(row.Hits > p25 and row.Hits <= p50))
    abv50p.append(int(row.Hits > p50 and row.Hits <= p75))
    abv75p.append(int(row.Hits > p75))

df = df.assign(num_paragraphs=num_paragraphs)
df = df.assign(num_pics=num_pics)
df = df.assign(text=text)
df = df.assign(num_words=num_words)
df = df.assign(num_chars=num_chars)
df = df.assign(num_tags=num_tags)
df = df.assign(abv75p=abv75p)
df = df.assign(abv50p=abv50p)
df = df.assign(abv25p=abv25p)
df = df.assign(bottom25p=bottom25p)

# Neural Net One: Multi Label
df1.insert(0, "text", text)
df1 = df1.assign(abv75p=abv75p)
df1 = df1.assign(abv50p=abv50p)
df1 = df1.assign(abv25p=abv25p)
df1 = df1.assign(bottom25p=bottom25p)

def removePunct(text):
    tokens = word_tokenize(text)
    words = [word for word in tokens if word.isalpha()]
    return " ".join(words)
    
def stemText(text):
    stemmer = SnowballStemmer("english")
    return ' '.join(stemmer.stem(word) for word in text.split())

#apply text cleaning methods
df1['text'] = df1['text'].str.lower().apply(removePunct).apply(stemText)

# Training The Model ----------------------------------------------------------
print("Training the model...")

train, test = train_test_split(df1, test_size=0.33, shuffle=True)
X_train = train.text
X_test = test.text

stop_words = set(stopwords.words('english'))

SVC_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words=stop_words)),
            ('clf', OneVsRestClassifier(LinearSVC(), n_jobs=1)),
        ])
    
tags = np.array((df1.columns.values)[1:])

totalAcc = 0
i = 0

AI_Assigned_Tags = []

for tag in tags:
    print('... Processing {}'.format(tag))
   
    #train the model using X_dtm & y
    SVC_pipeline.fit(X_train, train[tag])
    
    #compute the testing accuracy
    prediction = SVC_pipeline.predict(X_test)
    print('Test accuracy is {}'.format(accuracy_score(test[tag], prediction)))
    acc = accuracy_score(test[tag], prediction)
    
    totalAcc += acc 
    i += 1
    if i == 23:
        totalAcc = totalAcc / 24
        print("\n")
        print('Overall average test accuracy for tag predictions is {}'.format(totalAcc))
        totalAcc = 0
    if i == 27:
        totalAcc = totalAcc / 4
        print("\n")
        print('Overall average test accuracy for hit percentile predictions is {}'.format(totalAcc))
    
    finalPrediction = (SVC_pipeline.predict(df1.text))
    j = 0
    for elem in finalPrediction:
        if(j >= len(AI_Assigned_Tags)):
            AI_Assigned_Tags.append([])
        if (elem):
            AI_Assigned_Tags[j].append(tag)
        j += 1

df.to_csv('clean_data.csv')


