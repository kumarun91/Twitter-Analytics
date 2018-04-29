#Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import json

#Variables that contains the user credentials to access Twitter API 
access_token = "1543085354-k0HPvw9Hjhe0wrynEpcku3HHM88xSIGuJtYgaLa"
access_token_secret = "GD4ESj36zw5q00rNxiCCxjTOMr4HdSQzwjJ21HF93awgp"
consumer_key = "nnHxQWJw5W7zdUkjJyqL4qeHw"
consumer_secret = "8ZWgmzsI0v6oV4n37WXzn7yozDqmPN7CQemWciYCF0fkovJTpZ"


#This is a basic listener that just prints received tweets to stdout.
class StdOutListener(StreamListener):
    def __init__(self, api=None):
        super(StdOutListener, self).__init__()
        self.num_tweets = 0
    def on_data(self, data):
        textTweet = data
        data = json.loads(data)
        if 'lang' in data and data['lang'] == 'en' and ('DonaldTrump' in data['text'] or 'Donaldtrump' in data['text'] or 'Trump' in data['text']) and self.num_tweets < 1001:
            print('english trump tweet received ', data['text'])
            self.num_tweets += 1
            with open('WestTweets.txt', 'a') as file:
                file.write(textTweet)
            return True
        elif self.num_tweets >= 1000 :
            return False
        else:
            print 'Waiting for english trump tweets..'
            return True
    def on_error(self, status):
        print 'error ', status
if __name__ == '__main__':
    #This handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)

    #This line filter Twitter Streams to capture data by the keywords: 'python', 'javascript', 'ruby'
    stream.filter(locations = [-123.43,36.79,-113.33,48.65])
    