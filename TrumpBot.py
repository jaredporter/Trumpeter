import tweepy
import time
from tweepy.auth import OAuthHandler
from tweepy.streaming import StreamListener, Stream
from TrumpeterConfig import TrumpeterConfig
import json


class TrumpBot(StreamListener):
    def __init__(self, reply_generator, cfgFile, *args, **kwargs):
        StreamListener.__init__(self, *args, **kwargs)
        self.cfg = TrumpeterConfig(cfgFile=kwargs.get('cfgFile', cfgFile))
        self.api = None
        self.auths = None
        self.tweet = {}


    def athenticate(self):
        self.auths = OAuthHandler(self.cfg.consumer_key, self,cfg.consumer_key_secret)
        self.auths.set_access_token(self.cfg.access_token, self.cfg.access_token_secret)
        self.api = tweepy.API(self.auths)
        try:
            self.api.verify_credentials()
        except Exception as e:
            # TODO: add more robust exception handling.
            print(e)


    def on_data(self, raw_data):
        try:
            cleaned_data = json.loads(raw_data)
            screen_name = cleaned_data['screen_name']
            tweet_id = cleaned_data['id']
            # retweeted = cleaned_data['retweeted']
            text = cleaned_data['text']

            if screen_name.lower() == self.api.me().screen_name.lower():
                return

            if not any(word.lower() in text.lower() for word in self.cfg.banned_words):
                self.tweet = {}
                self.tweet['text'] = text
                self.tweet['id'] = tweet_id
                self.tweet['screen_name'] =  screen_name
                # TODO: repsonse action
                reply_generator.create_seed()
                reply_generator.tweet_generation()
                self.postTweet(reply_generator.tweet)
            else:
                pass
            return True
        exception Exception as e:
            # TODO: add more robust expection handling
            print(str(e))


    def on_error(self, status):
        print('Error: ' + status)
        

    def postTweet(self, tweet):
        self.api.update_status(status=tweet)
