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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances
from Reset_States_Callback import *
import os
import re
from collections import Counter


class Trumpeter(object):
    """
    Simple LSTM to generate tweets, character by character based on the
    ramblings of Prez Trump. Trump tweets were collected from the
    the Trump Twitter Archive: http://www.trumptwitterarchive.com
    Their repo is: https://github.com/bpb27/trump_tweet_data_archive
    """

    def __init__(self, filepath, batch_size=1028, hidden_layer_size=512,
            dropout=0.2, lr=0.005, decay=0.0, nb_epoch=10,
            stateful=False, continuation=False, max_seq=40, seq_step=3
            tweets_only=False):
        self.batch_size = batch_size
        self.hidden_layer_size = hidden_layer_size
        self.dropout = dropout
        self.lr = lr
        self.decay = decay
        self.nb_epoch = nb_epoch
        self.stateful = stateful
        self.continuation = continuation
        self.max_seq = max_seq
        self.seq_step = seq_step
        self.tweets_only = tweets_only
        self.seed = np.random.seed(42) 
        self.corpus = []
        self.generated_tweets = []
        self.filepath = filepath
        self.last_tweet = None
        self.chars = None
        self.char_to_idx = None
        self.idx_to_char = None
        self.n_chars = None
        self.corp_len = None
        self.seq_step = None
        self.max_seq = None
        self.sequences = []
        self.next_chars = []
        self.X = None
        self.y = None
        self.model = None


    def trump_loader(self):
        """
        Funciton to load in tweets into list. Each element is a tweet
        that's been clenaed and only the text has been pulled out.
        Also, it saves the most recent tweet ID so you know where you
        left off
        """
        # Read in tweets
        for file in os.listdir(self.filepath):
            if file.endswith(".txt") and self.tweets_only == False:
                with open(os.path.join(self.filepath, file)) as f:
                    self.corpus.append(f.read())
            elif file.endswith(".json"):
                with open(os.path.join(self.filepath, file)) as f:
                    twts = json.load(f)
        
                    # Remove all handles and urls and add it to the list of clean tweets
                    for text, entities in ((fix_text(t['text']), t['entities']) for t in twts):
                        urls = (e['url'] for e in entities['urls'])
                        users = ("@"+e['screen_name'] for e in entities['user_mentions'])
                        text = reduce(lambda t,s: t.replace(s, ''), chain(urls, users), text)
                        self.corpus.append(text)

        # # Remove tweets with sebsites in them. Use if above doesn't work
        # self.corpus = [t for t in self.corpus if 'http' not in t]
        puncs = ['/','(',')',':',';']
        self.corpus = [t for t in self.corpus if t not in puncs]

        # Convert list into single corpus
        self.corpus = ' '.join(self.corpus)
        self.corpus = re.sub('http://\S*', '', self.corpus)
        self.corp_len = len(self.corpus)

        self.last_tweet = twts[0]['created_at']


    def create_mappings(self):
        """
        Create the character maps
        """
        # # Count frequency of characters
        # counter = Counter(" ".join(self.corpus))
        # # Remove infrequent characters.
        # self.chars = [k for k, v in counter.items() if v > 9]

        # # Make sure that didn't remove standard characters
        # for i in string.ascii_letters:
        #     if i not in self.chars:
        #         self.chars.append(i)
        
        self.chars = sorted(list(set(" ".join(self.corpus))))
        # Number of unique characters
        self.n_chars = len(self.chars)
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        # Create the actual mappings
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}


    def sentence_creation(self):
        """
        Create the sentences for training. Keras docs use these params:
            max_seq = 40
            seq_step = 3
        Making steps or lengths shorter will result in longer training
        time, but on a GPU it shouldn't be prohibitively expensive.
        """
        for i in range(0, (self.corp_len - self.max_seq), self.seq_step):
            self.sequences.append(self.corpus[i:i + self.max_seq])
            self.next_chars.append(self.corpus[i + self.max_seq])
        self.n_seq = len(self.sequences)
        self.sequences = np.array(self.sequences)
        self.next_chars = np.array(self.next_chars)


    def one_hot_encode(self):
        """
        One hot encode the input and output so it's usable in a NN.
        """
        # Create empty matrices for Xs and ys
        self.X = np.zeros((self.n_seq, self.max_seq, self.n_chars), 
                dtype=np.bool)
        self.y = np.zeros((self.n_seq, self.n_chars), dtype=np.bool)
        # Loop through and update indicies of Xs and ys to one
        for i, sequence in enumerate(self.sequences):
            for t, char in enumerate(sequence):
                self.X[i, t, self.char_to_idx[char]] = 1
            self.y[i, self.char_to_idx[self.next_chars[i]]] = 1


    def model_creation(self, decay=0.0, stateful = False):
        """
        placeholder
        """
        self.hidden_layer_size = hidden_layer_size
        self.dropout = dropout
        self.lr = lr
        self.model = Sequential()
        if self.stateful == False:
            self.model.add(LSTM(self.hidden_layer_size,
                return_sequences=True, 
                input_shape = (self.max_seq, self.n_chars))) 
        else:
            self.model.add(LSTM(self.hidden_layer_size, 
                return_sequences=False, 
                batch_input_shape = (self.max_seq, self.n_chars),
                stateful = True)) 
        self.model.add(Dropout(self.dropout))
        if self.stateful == False:
            self.model.add(LSTM(self.hidden_layer_size,
                return_sequences=False, 
                input_shape = (self.max_seq, self.n_chars))) 
        else:
            self.model.add(LSTM(self.hidden_layer_size, 
                return_sequences=False, 
                batch_input_shape = (self.max_seq, self.n_chars),
                stateful = True)) 
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(self.n_chars, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(lr=self.lr, decay=self.decay))
       

    def train_model(self, batch_size = 1028, nb_epoch=7,
            hidden_layer_size = 512, dropout = 0.1, lr = 0.005,
            decay=0.0, continuation = False, max_seq = 40,
            seq_step = 3, stateful = False):
        """
        Train the model, obviously
        """
        try:
            if continuation == True:
                self.model.load_weights('weights.hdf5')
            checkpoint = ModelCheckpoint(filepath='weights.hdf5', 
                    monitor='loss', save_best_only=True, mode='min')
            self.model.fit(self.X, self.y,batch_size=batch_size, 
                    epochs=nb_epoch, callbacks=[checkpoint])

        except AttributeError:
            if not self.last_tweet:
                self.trump_loader()
            if not self.char_to_idx:
                self.create_mappings()
            if not self.sequences:
                self.sentence_creation()
            if not self.y:
                self.one_hot_encode()
            if not self.model:
                self.model_creation()

            if self.continuation == True:
                self.model.load_weights('weights.hdf5')
            checkpoint = ModelCheckpoint(filepath='weights.hdf5', 
                    monitor='loss', save_best_only=True, mode='min')
            resets = Reset_States_Callback()
            if self.stateful == True:
                self.model.fit(self.X, self.y,
                        batch_size=self.batch_size, 
                        epochs=self.nb_epoch, 
                        callbacks=[checkpoint, resets],
                        shuffle = False)
            else:
                self.model.fit(self.X, self.y,
                        self.batch_size=batch_size, 
                        epochs=self.nb_epoch, 
                        callbacks=[checkpoint, resets])


    def sample(self, preds):
        """
        This will give me the most likely character to occur next.
        """
        preds = np.asanyarray(preds).astype('float64') 
        preds = np.log(preds) / 0.2
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)


    def generate_tweets(self, seed, text_len, weights = 'weights.hdf5'):
        """
        This is where the magic happens. Generate tweets based on the
        sequences from the corpus. Eventually this will be able to take
        inputs from new tweets and generate things, but one thing at a
        time
        """
        try:
            # Load in our model's weights
            self.model.load_weights(weights)
            # # This finds a space in the corpus and then makes the seed from there
            # spaces_in_corpus = np.array([idx for idx in range(self.corp_len) 
            #     if self.corpus[idx] == ' '])
            # # Make the tweet, one letter at a time
            # begin = np.random.choice(spaces_in_corpus)
            tweet = u''
            # sequence = self.corpus[begin:begin + self.max_seq]
            sequence = seed
            tweet += sequence
            for _ in range(text_len):
                x = np.zeros((1, self.max_seq, self.n_chars))
                for t, char in enumerate(sequence):
                    x[0, t, self.char_to_idx[char]] = 1.0

                preds = self.model.predict(x, verbose = 0)[0]
                next_idx = self.sample(preds)
                next_char = self.idx_to_char[next_idx]

                tweet += next_char
                sequence = sequence[1:] + next_char
            return tweet

        except AttributeError:
            if not self.last_tweet:
                self.trump_loader()
            if not self.char_to_idx:
                self.create_mappings()
            if not self.sequences:
                self.sentence_creation()
            if not self.y:
                self.one_hot_encode()
            if not self.model:
                self.model_creation()
            # Load in our model's weights
            self.model.load_weights(weights)
            # This finds a space in the corpus and then makes the seed
            # from there
            spaces_in_corpus = np.array([idx for idx in
                range(self.corp_len) if self.corpus[idx] == ' '])
            # Make the tweet, one letter at a time
            # begin = np.random.choice(spaces_in_corpus)
            tweet = u''
            # sequence = self.corpus[begin:begin + self.max_seq]
            sequence = seed
            tweet += sequence
            for _ in range(text_len):
                x = np.zeros((1, self.max_seq, self.n_chars))
                for t, char in enumerate(sequence):
                    x[0, t, self.char_to_idx[char]] = 1.0

                preds = self.model.predict(x, verbose = 0)[0]
                next_idx = self.sample(preds)
                next_char = self.idx_to_char[next_idx]

                tweet += next_char
                sequence = sequence[1:] + next_char
            return tweet


    def evaluation(self):
        """
        Evaluate the cosine distance between generated tweets and the
        sequences from the corpus.
        """
        vectoriser = TfidfVectorizer()
        tfidf = vectoriser.fit_transform(self.sequences)
        Xval = vectorizer.transform(self.generated_tweets)
        print(pairwise_distances(Xval, Y=tfidf, metric='cosine').min(
            axis=1).mean())
