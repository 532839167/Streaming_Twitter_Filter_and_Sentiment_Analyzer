import re
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

class TweetAnalyzer:

    emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad',
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed',
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink',
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

    def __init__(self):
        # Load the vectoriser.
        file = open('/Users/songjin/Desktop/proj/filter/tweet-sentiment-kaggle/vectoriser-ngram-(1,2).pickle', 'rb')
        self.vectoriser = pickle.load(file)
        file.close()

        # Load the LR Model.
        file = open('/Users/songjin/Desktop/proj/filter/tweet-sentiment-kaggle/Sentiment-LR.pickle', 'rb')
        self.model = pickle.load(file)
        file.close()

    def preprocess(self, textdata):
        processedText = []

        # Create Lemmatizer and Stemmer.
        wordLemm = WordNetLemmatizer()

        # Defining regex patterns.
        urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
        userPattern       = '@[^\s]+'
        alphaPattern      = "[^a-zA-Z0-9]"
        sequencePattern   = r"(.)\1\1+"
        seqReplacePattern = r"\1\1"

        stop_w=stopwords.words("english")

        for tweet in textdata:
            tweet = tweet.lower()

            # Replace all URls with 'URL'
            tweet = re.sub(urlPattern,' URL',tweet)
            # Replace all emojis.
            for emoji in self.emojis.keys():
                tweet = tweet.replace(emoji, "EMOJI" + self.emojis[emoji])
            # Replace @USERNAME to 'USER'.
            tweet = re.sub(userPattern,' USER', tweet)
            # Replace all non alphabets.
            tweet = re.sub(alphaPattern, " ", tweet)
            # Replace 3 or more consecutive letters by 2 letter.
            tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

            tweetwords = ''
            for word in tweet.split():
                # Checking if the word is a stopword.
                #if word not in stopwordlist:
                if len(word)>1 and word not in stop_w:
                    # Lemmatizing the word.
                    word = wordLemm.lemmatize(word)
                    tweetwords += (word+' ')

            processedText.append(tweetwords)

        return processedText

    def predict(self, text):
        # Predict the sentiment
        text_list = []
        text_list.append(text)
        textdata = self.vectoriser.transform(self.preprocess(text_list))
        sentiment = self.model.predict(textdata)

        return int(sentiment[0])