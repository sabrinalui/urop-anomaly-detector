import tweepy

class MyStreamListener(tweepy.StreamListener):

	def on_status(self, status):
		print(status.text)