from tweet_analyzer import TweetAnalyzer
from textblob import TextBlob

example = "Bad bad! Very bad!"

analyzer = TweetAnalyzer()
blob = TextBlob(example)
sent = blob.sentiment

print(analyzer.predict(example))
print(sent.polarity)