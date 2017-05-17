from lstm_tweet_generator import Trumpeter
from nltk import pos_tag
from string import punctuation
import enchant
import re

class create_reply(object):
    """
    This is will allow us to take in the tweet Trumpeter will reply to, 
    pull out handles, pick a noun as the seed text, and then generate 
    the tweet.
    """

    def __ini__(self, Trumpeter):
        self.initial_tweet = initial_tweet
        self.vocab = vocab
        self.replying_to = []
        self.hashtags = None
        self.seed = None
        self.tweet = None
        self.trumpeter = Trumpeter


    def create_seed(self):
        # Save the handle we're replying to
        author = self.initial_tweet['user']['screen_name']
        author = "@" + author 
        self.replying_to.append(author)
        self.hashtags = self.initial_tweet['entities']['hashtags']
        # Get the text of the tweet we're replying to
        text = self.initial_tweet['text']
        handles_regex = re.compile(r"(?<=^|(?<=[^a-zA-Z0-9-_\.]))(@[A-Za-z]+[A-Za-z0-9]+)")
        for h in handles_regex.findall(text):
            self.replying_to.append(h)
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
            if ( s_tag in parts 
                    and s_lower_tag in parts 
                    and (s_tag in vocab 
                         or s_lower_tag in vocab)
                ):
                tagged.append(s_tag)
        # Randomly pick a noun and the following word to start our reply
        if len(tagged) > 0:
            self.seed = random.choice(tagged)
            self.seed += ' ' + self.initial_tweet['text'].partition(
                    self.seed).split()[0]
    
    
    def tweet_generation(self):
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
        # Pick a hashtag to use from those in original tweet
        try:
            hashtag = random.choice(self.hashtags)
            hashtag = "#" + hashtag['text']
        except IndexError:
            hashtag = ''
        # Take that charcter count off of the limit
        remaining -= len(first_bit)
        remaining -= len(self.seed)
        remaining -= len(hashtag)
        # Create the tweet 
        # TODO: check to make sure seed is long enough to generate 
        # a tweet that makes sense
        self.tweet = self.trumpeter.generate_tweets(self.seed, remaining)
        # Put it all together
        self.tweet = first_bit + self.tweet
        spell_check = enchant.Dict("en_us")
        last_word = self.tweet.split()[-1]) 
        if spell_check.check(last_word): 
            self.tweet += hashtag
        else:
            corrected = spell_check.suggest(last_word)[0]
            temp_tweet = ' '.join(self.tweet.split()[:-1] + list(corrected))
            if len(temp_tweet + ' ' + hashtag) < 141:
                self.tweet = temp_tweet + ' ' + hashtag
            else:
                self.tweet = ' '.join(self.tweet.split()[:-1] + ' '+ hashtag)
