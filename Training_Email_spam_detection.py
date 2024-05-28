import pandas as pd
import numpy as np
import nltk
import string

from nltk.corpus import stopwords

data=pd.read_csv("D:\MY_PROJ\emails_1.csv")
data = data.iloc[:,:2]
print(data.head())

print(data.describe().T)

print(data.shape)

df = data.dropna()
print(data.isnull().sum())

df.drop_duplicates(inplace=True)
print(df.shape)
print(data['spam'].value_counts())

nltk.download('stopwords')

punc = string.punctuation
punc_list = list(punc)
print(punc_list)


def process_text(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    clean_words = [word for word in nopunc.split() if word.lower() not in punc_list]

    return clean_words

print(df['text'].head().apply(process_text))

from sklearn.feature_extraction.text import CountVectorizer
messages_bow=CountVectorizer(analyzer=process_text).fit_transform(df['text'])

print(messages_bow.dtype)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(messages_bow, df['spam'], random_state=42, test_size=0.20)

print(messages_bow.shape)

print(X_train.dtype)

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB().fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

pred = classifier.predict(X_test)
print(classification_report(y_test, pred))

print(confusion_matrix(y_test, pred))

print(classifier.score(X_test,y_test))
