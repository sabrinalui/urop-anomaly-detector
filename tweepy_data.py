import tweepy
import yaml
from stream_listener import MyStreamListener
from oauth import CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET

auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

api = tweepy.API(auth)

my_stream_listener = MyStreamListener()
my_stream = tweepy.Stream(auth = api.auth, listener=my_stream_listener)
my_stream.filter(track=['sabrina'])