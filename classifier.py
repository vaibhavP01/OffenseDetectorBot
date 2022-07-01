import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.pipeline import Pipeline

#to data preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

#NLP tools
import re
import nltk
nltk.download('stopwords')
nltk.download('rslp')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

#train split and fit models
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from nltk.tokenize import TweetTokenizer

#model selection
from sklearn.metrics import confusion_matrix, accuracy_score, plot_confusion_matrix, classification_report

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

dataset = pd.read_csv('labeled_data.csv')
#print(dataset.head())

dt_transformed = dataset[['class', 'tweet']]
y = (dt_transformed.iloc[:, :-1].values).ravel()
#print(dt_transformed.head())

#Splitting df into training and testing
df_train, df_test = train_test_split(dt_transformed, test_size = 0.10, random_state = 42, stratify=dt_transformed['class'])
#print(df_train.shape, df_test.shape)

#Splitting df into training and validation
df_train, df_vad = train_test_split(df_train, test_size = 0.10, random_state = 42, stratify=df_train['class'])
#print(df_train.shape, df_vad.shape)

df_train['class'].value_counts().plot(kind='bar')

def preprocessing(data):
    stemmer = nltk.stem.RSLPStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    corpus = []
    for tweet in data:
      review = re.sub(r"@[A-Za-z0-9_]+", " ", tweet)
      review = re.sub('RT', ' ', review)
      review = re.sub(r"https?://[A-Za-z0-9./]+", " ", review)
      review = re.sub(r"https?", " ", review)
      review = re.sub('[^a-zA-Z]', ' ', review)
      review = review.lower()
      review = review.split()
      ps = PorterStemmer()
      review = [ps.stem(word) for word in review if not word in set(all_stopwords) if len(word) > 2]
      review = ' '.join(review)
      corpus.append(review)

    return np.array(corpus)

c_train = preprocessing(df_train['tweet'].values)
c_vad = preprocessing(df_vad['tweet'].values)

####Extracting features using tokenization####

tweet_tokenizer = TweetTokenizer()
vectorizer = CountVectorizer(analyzer="word", tokenizer=tweet_tokenizer.tokenize, max_features=1010)


def tokenize(corpus, flag=0):
    # flag = 1 --> treino
    if (flag):
        return vectorizer.fit_transform(corpus).toarray()
    else:
        return vectorizer.transform(corpus).toarray()

X_train = tokenize(c_train, 1)
X_vad = tokenize(c_vad, 0)
y_train = df_train['class'].values
y_vad = df_vad['class'].values
X_train.shape, X_vad.shape

# Logistic Regression
model = LogisticRegression(multi_class='ovr', solver='liblinear', random_state = 0)
model.fit(X_train, y_train.ravel())
y_pred = model.predict(X_vad)

###Making the Confusion Matrix##
def set_confusion_matrix(clf, X, y, title):
    plot_confusion_matrix(clf, X, y)
    plt.title(title)
    plt.show()

set_confusion_matrix(model, X_vad, y_vad, type(model).__name__)

target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_vad, y_pred, target_names=target_names))

###Analyzing better the words of each class###

conjunto = c_train
hate_tweets = [sentence for sentence, label in zip(conjunto, y) if label == 0]
off_tweets = [sentence for sentence, label in zip(conjunto, y) if label == 1]
none_tweets = [sentence for sentence, label in zip(conjunto, y) if label == 2]

hate_words = ' '.join(hate_tweets)
off_words = ' '.join(off_tweets)
none_words = ' '.join(none_tweets)


###         wordcloud       ###
def get_wordcloud(text):
    # Create and generate a word cloud image:
    wordcloud = WordCloud().generate(text)

    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

get_wordcloud(hate_words)
get_wordcloud(off_words)
get_wordcloud(none_words)

def wordListToFreqDict(wordlist):
    wordfreq = [(wordlist.count(p))/len(wordlist) for p in wordlist]
    return dict(list(zip(wordlist,wordfreq)))

def sortFreqDict(freqdict):
    aux = [(freqdict[key], key) for key in freqdict]
    aux.sort()
    aux.reverse()
    return aux

hate_dict = sortFreqDict(wordListToFreqDict(hate_words.split()))
off_dict = sortFreqDict(wordListToFreqDict(off_words.split()))
none_dict = sortFreqDict(wordListToFreqDict(none_words.split()))

len(hate_dict), len(off_dict), len(none_dict)

###     Catching the words that appear most in each class      ######

def get_common(wordlist, n):
    return ([w[1] for w in wordlist])[:n]

common_words = list()
common_words.append(get_common(hate_dict, 2000))
common_words.append(get_common(off_dict, 1000))
common_words.append(get_common(none_dict, 1000))
common_words = np.unique(np.hstack(common_words))

common_words_dict = ({i:j for i, j in zip(common_words, range(len(common_words)))})

X_train = tokenize(c_train, 1)
X_vad = tokenize(c_vad, 0)
X_train.shape, X_vad.shape

# Logistic Regression
model = LogisticRegression(multi_class='ovr', solver='liblinear', random_state = 0)
model.fit(X_train, y_train.ravel())
y_pred = model.predict(X_vad)

set_confusion_matrix(model, X_vad, y_vad, type(model).__name__)

target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_vad, y_pred, target_names=target_names))

n_off, n_none, n_hate = df_train['class'].value_counts()
#print(n_hate, n_off, n_none)

df_hate = df_train[df_train['class'] == 0]
df_off = df_train[df_train['class'] == 1]
df_none = df_train[df_train['class'] == 2]

df_off_under = df_off.sample(n_hate, random_state=0)
df_none_under = df_none.sample(n_hate, random_state=0)

df_under = pd.concat([df_hate, df_off_under, df_none_under], axis=0)
print(df_under['class'].value_counts())

####Now training the models with this data:###

c_train = preprocessing(df_under['tweet'].values)
c_vad = preprocessing(df_vad['tweet'].values)

X_train = tokenize(c_train, 1)
X_vad = tokenize(c_vad, 0)
y_train = df_under['class'].values
y_vad = df_vad['class'].values
X_train.shape, X_vad.shape

# Logistic Regression
model_under = LogisticRegression(multi_class='ovr', solver='liblinear', random_state = 0)
model_under.fit(X_train, y_train.ravel())
y_pred = model_under.predict(X_vad)

set_confusion_matrix(model_under, X_vad, y_vad, type(model_under).__name__)

target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_vad, y_pred, target_names=target_names))

####            OVERSAMPLING            ####

n_off, n_none, n_hate = df_train['class'].value_counts()
print(n_hate, n_off, n_none)

df_hate_over = df_hate.sample(n_off, replace=True, random_state=0)
df_none_over = df_none.sample(n_off, replace=True, random_state=0)
df_over = pd.concat([df_off, df_hate_over, df_none_over], axis=0)

print('Random over-sampling:')
print(df_over['class'].value_counts())

c_train = preprocessing(df_over['tweet'].values)
c_vad = preprocessing(df_vad['tweet'].values)

X_train = tokenize(c_train, 1)
X_vad = tokenize(c_vad, 0)
y_train = df_over['class'].values
y_vad = df_vad['class'].values
X_train.shape, X_vad.shape

# Logistic Regression
model_over = LogisticRegression(multi_class='ovr', solver='liblinear', random_state = 0)
model_over.fit(X_train, y_train.ravel())
y_pred = model_over.predict(X_vad)

set_confusion_matrix(model_over, X_vad, y_vad, type(model_over).__name__)
target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_vad, y_pred, target_names=target_names))


#####       Test data preview           ####

c_test = preprocessing(df_test['tweet'].values)

print(c_test.shape)

X_test = tokenize(c_test, 0)
y_test = df_test['class']

print(X_test.shape, y_test.shape)
y_pred = model_over.predict(X_test)
set_confusion_matrix(model_over, X_test, y_test, type(model_over).__name__)
target_names = ['class 0', 'class 1', 'class 2']
print("FINAL--- FINAL ----FINAL ----FINAL")
print("HELLO FROM THE OTHER SIDE")
print(classification_report(y_test, y_pred, target_names=target_names))

'''
0 - Hate speech

1 - Offensive language

2 - neither
'''