from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import numpy as np
import random
import sys
import json
from ftfy import fix_text
import string
from itertools import chain
from six.moves import reduce


class Trumpeter(filepath):
    """
    Simple LSTM to generate tweets, character by character based on the
    ramblings of Prez Trump. Trump tweets were collected from the
    the Trump Twitter Archive: http://www.trumptwitterarchive.com
    """

    def __init__:
        self.tweets = []
        self.filepath = filepath
        self.last_tweet = None
        self.chars = None
        self.char_map = None


    def trump_loader(self):
        """
        Funciton to load in tweets into list. Each element is a tweet
        that's been clenaed and only the text has been pulled out.
        Also, it saves the most recent tweet ID so you know where you
        left off
        """
        # Read in tweets
        with open(self.filepath) as f:
            twts = json.load(f)
        
        # Remove all handles and urls and add it to the list of clean tweets
        for text, entities in ((fix_text(t['text']), t['entities']) for t in tweets):
            urls = (e['url'] for e in entities['urls'])
            users = ("@"+e['screen_name'] for e in entities['user_mentions'])
            text = reduce(lambda t,s: t.replace(s, ''), chain(urls, users), text)
            self.tweets.append(text)

        self.last_tweet = twts[0]['created_at']


    def data_prep(self):
        counter = Counter(" ".join(self.tweets))
        self.chars = [k for k, v in counter.items() if v > 9]
        for i in string.ascii_letters:
            if i not in self.chars:
                self.chars.append(i)
        self.chars = sorted(self.chars)
