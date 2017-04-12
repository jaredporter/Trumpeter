from lstm_tweet_generator import Trumpeter
from nltk import pos_tag
from string import punctuation

def create_seed(initial_tweet):
    # Save the handle we're replying to
    replying_to = initial_tweet['user']['screen_name']
    replying_to = "@" + replying_to
    # Get the text of the tweet we're replying to
    text = initial_tweet['text']
    # Tokenise the text and remove handles, hashtags, and urls
    text = [s.translate(str.maketrans('','',punctuation)) for s in 
            text.split() if not s.startswith(('#','@','http'))]
    text = [s for s in text if 'trump' not in s.lower() and s.isalpha()]
    # Do parts of speech tagging
    parts = ['NNS', 'NNP', 'NN', 'NNPS']
    tagged = []
    # Create a list of nouns to pick as a seed.
    for s in text:
        s_tag = pos_tag([s])[0]
        s_lower_tag = pos_tag([s.lower()])[0]
        if s_tag in parts and s_lower_tag in parts:
            tagged.append(s_tag)
    # Randomly pick a noun to start our reply
    seed = random.choice(tagged)
    return seed
