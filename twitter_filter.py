import tweepy
import datetime
from tweet_analyzer import TweetAnalyzer
from tweet_store import TweetStore
import json

file_path = 'config/api.json'

with open(file_path) as f:
    twitter_api = json.loads(f.read())

consumer_key = twitter_api['consumer_key']
consumer_secret = twitter_api['consumer_secret']
access_token = twitter_api['access_token']
access_token_secret = twitter_api['access_token_secret']

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)
store = TweetStore()

class StreamListener(tweepy.StreamListener):

    def on_status(self, status):

        if ('RT @' not in status.text):
            analyzer = TweetAnalyzer()
            polarity = analyzer.predict(status.text)
            # polarity = sent.polarity
            # subjectivity = sent.subjectivity

            tweet_item = {
                'id_str': status.id_str,
                'text': status.text,
                'polarity': polarity,
                # 'subjectivity': subjectivity,
                'username': status.user.screen_name,
                'name': status.user.name,
                'profile_image_url': status.user.profile_image_url,
                'received_at': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            store.push(tweet_item)
            print("Pushed to redis:", tweet_item)

    def on_error(self, status_code):
        if status_code == 420:
            return False

stream_listener = StreamListener()
stream = tweepy.Stream(auth=api.auth, listener=stream_listener)
stream.filter(track=["@Facebook", "@Microsoft", "@Apple", "@Google", "@Amazon", "@Netfilix"])
