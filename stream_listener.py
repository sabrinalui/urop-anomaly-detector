import tweepy

class TwitterStreamListener(tweepy.StreamListener):

	def on_status(self, status):
		print(status.text)