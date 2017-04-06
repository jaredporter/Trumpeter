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
        self.char_to_idx = None
        self.idx_to_char = None
        self.N_CHARS = None
        self.corp_len


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

        # # Remove tweets with sebsites in them. Use if above doesn't work
        # self.tweets = [t for t in self.tweets if 'http' not in t]

        # Convert list into single corpus
        self.tweets = ' '.join(self.tweets)
        self.corp_len = len(self.tweets)

        self.last_tweet = twts[0]['created_at']


    def create_mappings(self):
        """
        Create the character maps
        """
        # Count frequency of characters
        counter = Counter(" ".join(self.tweets))
        # Remove infrequent characters.
        self.chars = [k for k, v in counter.items() if v > 9]

        # Make sure that didn't remove standard characters
        for i in string.ascii_letters:
            if i not in self.chars:
                self.chars.append(i)
        
        self.chars = sorted(self.chars)
        # Number of unique characters
        self.N_CHARS = len(self.chars)
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        # Create the actual mappings
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}


    def sentence_creation(self, max_seq, seq_step, n_seq):
        """
        Create the sentences for training. Keras docs use these params:
            seq_len = 40
            seq_step = 3
        Making steps or lengths shorter will result in longer training
        time, but on a GPU it shouldn't be prohibitively expensive.
        """

        sequences, next_chars = [], []
        for i in range(0, (self.corp_len - seq_len), seq_step):
            sequences.append(self.tweets[i:i + seq_len])
            next_chars.append(self.tweets[i + seq_len])
