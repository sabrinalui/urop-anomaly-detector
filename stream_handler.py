import tweepy
import time
import re
import pickle
from model import MLEModel
from collections import defaultdict

# English stopwords taken from NLTK
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']

class StreamingAnomalyHandler(tweepy.StreamListener):

	def __init__(self, t0, t1):
		self.t0 = t0
		self.t1 = t1
		self.models = {}
		self.tweet_buffer = defaultdict(list)
		self.last_time = int(time.time())
		super(StreamingAnomalyHandler, self).__init__()

	def on_status(self, status):
		(tweet,loc) = self.sanitize(status)
		if loc[0] != None and loc[1] != None:
			curr_time = int(time.time())
			self.tweet_buffer[loc].append(tweet)
			if (curr_time - self.last_time > self.t1):
				for loc,tweets in self.tweet_buffer.items():
					if loc not in self.models:
						self.models[loc] = MLEModel(loc)
					self.models[loc].update_priors(tweets,curr_time-self.t0)
					self.models[loc].update_posteriors(tweets,curr_time-self.t1)

				self.tweet_buffer = defaultdict(list)
				self.last_time = curr_time
				self.write_models(self.models)
				print("Model priors and posteriors updated at {}.".format(time.strftime("%H:%M:%S",time.localtime(curr_time))))

	def sanitize(self, status):
		body_words = status.text.split()
		timestamp = int(status.created_at.timestamp())
		filtered_words = [word for word in body_words if word.lower() not in stop_words]
		body = ' '.join(filtered_words)
		body = re.sub(r"http\S+", "<URL>",body)
		loc = (status.user.lang,status.user.time_zone)
		return ((body,timestamp),loc)

	def write_models(self, models):
		pickle.dump(models, open("modeldump.pkl","wb"))