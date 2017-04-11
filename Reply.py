from lstm_tweet_generator import Trumpeter
from nltk import word_tokenize, pos_tag

def create_reply(initial_tweet):
    replying_to = initial_tweet['user']['screen_name']
    replying_to = "@" + replying_to
    text = initial_tweet['text']
    text = word_tokenize(text)
    text = [s for s in text if 'trump' not in s.lower() and s.isalpha()]
    parts = ['NNS', 'NNP', 'NN', 'NNPS']
    tagged = []
    for s in text:
        s_tag = pos_tag([s])[0]
        s_lower_tag = pos_tag([s.lower()])[0]
        if s_tag in parts and s_lower_tag in parts:
            tagged.append(s_tag)
