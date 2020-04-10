import pandas as pd
import numpy as np
import pickle

import re
import os
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from __future__ import print_function
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import RMSprop, Adam
from keras.utils.data_utils import get_file
from tensorflow.keras.models import load_model

from flask import Flask,render_template,url_for,request


app = Flask(__name__)
#logreg = pickle.load(open('logreg_model.pkl', 'rb'))

#regex to clean text
#remove empty lines
#repalce '\n' with ' \n' so that we can treat the new line charchter as a seperate word
def clean(row):
    row = re.sub(r'[\(\[].*?[\)\]]', '', row)
    row = os.linesep.join([s for s in row.splitlines() if s])
    row = row.lower().replace('\n', ' \n ')
    return row

#Reads a csv file given the path as a parameter
def readFile(path):
  df = pd.read_csv(path)
  return dfs

#Sentiment Analysis using nltk vader
sia = SentimentIntensityAnalyzer()

def sentiment(row):
    num_positive = 0
    num_negative = 0
    num_neutral = 0
    sentences = row['text'].splitlines()
    for sentence in sentences:
        comp = sia.polarity_scores(sentence) 
        comp = comp['compound']
        if comp >= 0.5:
            num_positive += 1
        elif comp > -0.5 and comp < 0.5:
            num_neutral += 1
        else:
            num_negative += 1
    num_total = num_negative + num_neutral + num_positive
    percent_negative = (num_negative/float(num_total))*100
    percent_neutral = (num_neutral/float(num_total))*100
    percent_positive = (num_positive/float(num_total))*100
    row['pos'] = percent_positive
    row['neg'] = percent_negative
    row['neu'] = percent_neutral
    return row

def UserLyricssentiment(row):
    num_positive = 0
    num_negative = 0
    num_neutral = 0
    sentences = row['text'].splitlines()
    for sentence in sentences:
        comp = sia.polarity_scores(sentence) 
        comp = comp['compound']
        if comp >= 0.5:
            num_positive += 1
        elif comp > -0.5 and comp < 0.5:
            num_neutral += 1
        else:
            num_negative += 1
    num_total = num_negative + num_neutral + num_positive
    percent_negative = (num_negative/float(num_total))*100
    percent_neutral = (num_neutral/float(num_total))*100
    percent_positive = (num_positive/float(num_total))*100
    #row['pos'] = percent_positive
    #row['neg'] = percent_negative
    #row['neu'] = percent_neutral
    if(percent_positive < percent_negative):
        return 2    
    return 1

#Creates a seperate dataframe for positive artist with their lyrics
#num paramter is the number of artist that you want to select
def posArtist(num):
  positive_artists = positiveArt['artist'][:num].to_frame()
  temp_lyrics = pd.merge(df, positive_artists, how='inner', on = 'artist')
  temp_lyrics = temp_lyrics.sample(frac=1).reset_index(drop=True)
  arts_list = temp_lyrics['artist'].unique()
  print(len(arts_list))
  temp_lyrics.head()
  return temp_lyrics

#Creates a seperate dataframe for negative artist with their lyrics
#num paramter is the number of artist that you want to select
def negArtist(num):
  negative_artists = negativeArt['artist'][:num].to_frame()
  temp_lyrics = pd.merge(df, negative_artists, how='inner', on = 'artist')
  temp_lyrics = temp_lyrics.sample(frac=1).reset_index(drop=True)
  arts_list = temp_lyrics['artist'].unique()
  print(len(arts_list))
  temp_lyrics.head()
  return temp_lyrics

#Creates a seperate dataframe for neutral artist with their lyrics
#num paramter is the number of artist that you want to select
def neuArtist(num):
  neutral_artists = neutralArt['artist'][:num].to_frame()
  temp_lyrics = pd.merge(df, neutral_artists, how='inner', on = 'artist')
  temp_lyrics = temp_lyrics.sample(frac=1).reset_index(drop=True)
  arts_list = temp_lyrics['artist'].unique()
  print(len(arts_list))
  temp_lyrics.head()
  return temp_lyrics

#Creates a corpus after using the dataframe returned by posArtist(), negArtist() or neuArtist()
def createCorpus(temp_lyrics):
  corpus = temp_lyrics['text'].str.cat(sep='\n').lower()
  print('corpus length:', len(corpus))
  chars = sorted(list(set(corpus)))
  print('total chars:', len(chars))
  return chars, corpus

#Create a 2-way mapping of every character to a unique id (and unique id to a character)
def lstmPrep(chars, corpus):
  char_indices = dict((c, i) for i, c in enumerate(chars))
  indices_char = dict((i, c) for i, c in enumerate(chars))

  #The window size
  maxlen = 40
  #The steps between the windows 
  step = 3 
  sentences = []
  next_chars = []

  #Take a sequence of 40 characters at a time and a step size of 3 to create the overlapping samples 
  for i in range(0, len(corpus) - maxlen, step):
      sentences.append(corpus[i: i + maxlen]) 
      next_chars.append(corpus[i + maxlen])
  sentences = np.array(sentences)
  next_chars = np.array(next_chars)
  print('Number of sequences:', len(sentences))
  return char_indices, indices_char, maxlen, step, sentences, next_chars

#Helper function to generate character level one-hot encoding for each sentence and next_character
def getdata(sentences, next_chars):
    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    length = len(sentences)
    index = 0
    for i in range(len(sentences)):
        sentence = sentences[i]
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
    return X, y

#This function is used to add a bit of variance in the selection
#Without this it will keep printing the same lines over and over
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    #print(preds)
    probas = np.random.multinomial(1, preds, 1)
    #print(probas)
    return np.argmax(probas)


df = readFile('/content/drive/My Drive/nlpProjectdata/songdata.csv')
df['text'] = df['text'].apply(clean)
cleanLyrics = df.groupby(['artist'])['text'].apply(lambda x: '\n'.join(x)).reset_index()
cleanLyrics = cleanLyrics.apply(sentiment, axis=1)
cleanLyrics.reset_index()
#Seperates artists by postive, negative and neutral sentiments
positiveArt = cleanLyrics.sort_values(by=['pos'],ascending = False)
negativeArt = cleanLyrics.sort_values(by=['neg'],ascending = False)
neutralArt = cleanLyrics.sort_values(by=['neu'],ascending = False)
temp_lyrics = None
model = Sequential()
InputLyrics = None
my_prediction = 0
userInput = 0
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        if (request.form['submit_button'] == 'User input lyrics'):
            InputLyrics = request.form['InputLyrics']
            my_prediction = UserLyricssentiment(InputLyrics)
            userInput = 1
        if (request.form['submit_button'] == 'Auto generate positive input lyrics' or my_prediction == 1):
            temp_lyrics = posArtist(num)
            model = load_model('positive_songs_generate.h5')
            my_prediction = 1
        if (request.form['submit_button'] == 'Auto generate negative input lyrics' or my_prediction == 2):
            temp_lyrics = negArtist(num)
            model = load_model('negative_songs_generate.h5')
            my_prediction = 2
        chars, corpus = createCorpus(temp_lyrics)
        char_indices, indices_char, maxlen, step, sentences, next_chars = lstmPrep(chars, corpus)
        if(userInput == 0):
            sentence = df.iloc[np.random.random_integers(len(temp_lyrics.index))]['text'].lower()[:40]
        elif(userInput == 1):
            sentence = InputLyrics
        x = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1.
        variance = 0.50
        generated = ''
        original = sentence
        window = sentence
        # Predict the next 400 characters based on the seed
        for i in range(400):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(window):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
    
            next_index = sample(preds, variance) 
    
            next_char = indices_char[next_index]

            generated += next_char
            window = window[1:] + next_char
    return render_template('result.html',prediction = my_prediction, lyrics = original + generated)

if __name__ == '__main__':
    app.run(debug=True)