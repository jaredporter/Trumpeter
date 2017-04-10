import tweepy
import time
from tweepy.auth import OAuthHandler
from tweepy.streaming import StreamListener, Stream
from TrumpeterConfig import TrumpeterConfig


class Trumpeter(StreamListener):
    def __init__(self,cfgFile, *args, **kwargs):
        StreamListener.__init__(self, *args, **kwargs)
        self.cfg = TrumpeterConfig(cfgFile=kwargs.get('cfgFile', cfgFile))
        self.api = None
        self.auths = None


    def athenticate(self):
        self.auths = OAuthHandler(self.cfg.consumer_key, self,cfg.consumer_key_secret)
        self.auths.set_access_token(self.cfg.access_token, self.cfg.access_token_secret)
        self.api = tweepy.API(self.auths)
        try:
            self.api.verify_credentials()
        except Exception as e:
            print(e)
