import os
import ConfigParser
import inspect
import hashlib
import json


class TrumpeterConfig(object):
    def __init__(self, cfgFile='config'):
        path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

        self.config = ConfigParser.SafeConfigParser()
        self.config.read(os.path.join(path, 'config'))

        self.consumer_key = self.config.get("twitter", "consumer_key")
        self.consumer_key_secret = self.config.get("twitter", "consumer_secret")
        self.access_token = self.config.get("twitter", "access_token")
        self.access_token_secret = self.config.get("twitter", "access_token_secret")

        self.track_words = json.loads(self.config.get("track","words"))
        self.track_account = json.loads(self.config.get("track","account"))

        self.banned_accounts = json.loadds(self.config.get("banned","accounts"))
        self.banned_words = json.loads(self.config.get("banned","words"))
        
        self.whitelist_accounts = json.loads(self.config.get("whitelist","accounts"))
        self.whitelist_words = json.loads(self.config.get("whitelist","words"))

        self.follow_accounts = json.loads(self.config.get("follow","accounts"))
