from lstm_tweet_generator import Trumpeter
from nltk import pos_tag
from string import punctuation

class create_reply(object):
    """
    This is will allow us to take in the tweet Trumpeter will reply to, 
    pull out handles, pick a noun as the seed text, and then generate 
    the tweet.
    """

    def __ini__(self):
        self.initial_tweet = initial_tweet
        self.replying_to = None
        self.seed = None
        self.tweet = None

    def create_seed(self):
        # Save the handle we're replying to
        self.replying_to = self.initial_tweet['user']['screen_name']
        self.replying_to = "@" + self.replying_to
        # Get the text of the tweet we're replying to
        text = self.initial_tweet['text']
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
        self.seed = random.choice(tagged)
    
    
    def tweet_generation(self, LSTM):
        # TODO: Make sure the tweets end with a complete word/punctuation
        # Set the character limit
        remaining = 137
        # Create the . plus handles we're responding to text
        if type(self.replying_to) == list:
            first_bit = '.' + ' '.join(self.replying_to)
        else:
            first_bit = '.' + self.replying_to
        # Add the seed text to that
        first_bit += " "
        # Take that charcter count off of the limit
        remaining -= len(first_bit)
        remaining -= len(self.seed)
        # Create the tweet
        self.tweet = LSTM.generate_tweets(self.seed, remaining)
        # Put it all together
        self.tweet = first_bit + self.tweet
