from lstm_tweet_generator import *
from Reply import *
from TrumpBot import *

if __name__ == '__main__':
    tr = Trumpeter()
    cr = create_reply(tr)
    connection = TrumpBot('config')
