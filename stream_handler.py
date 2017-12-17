import tweepy
import ipdb
import pytz
from datetime import datetime
import re
import pickle
from model import MLEModel
from collections import defaultdict
from threading import Timer,Thread,Event
from sklearn.feature_extraction.text import CountVectorizer

# English stopwords taken from NLTK
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']

class StreamingAnomalyHandler(tweepy.StreamListener):

	def __init__(self):
		self.tweet_buffer = defaultdict(list)

		super(StreamingAnomalyHandler, self).__init__()

	def on_status(self, status):
		if status.user.lang == "en":
			(tweet,loc) = self.sanitize(status)
			if loc != None:
				self.tweet_buffer[loc].append(tweet)
		# if loc != None:
		# 	self.tweet_buffer[loc.country].append(tweet)

	def sanitize(self, status):
		body_words = status.text.split()
		timestamp = int(status.created_at.astimezone(pytz.utc).timestamp())
		filtered_words = [word for word in body_words if word.lower() not in stop_words]
		body = ' '.join(filtered_words)
		body = re.sub(r"http\S+", "<URL>",body)
		loc = status.user.time_zone
		# loc = status.place
		return ((body,timestamp),loc)

	def get_buffer(self):
		return self.tweet_buffer

	def clear_buffer(self):
		self.tweet_buffer = defaultdict(list)


class UpdaterThread(Thread):

	def __init__(self,event,t0,t1,handler):
		Thread.__init__(self)
		self.stopped = event
		self.models = {}
		self.t0, self.t1 = t0,t1
		self.handler = handler
		self.a = 0

	def run(self):
		while not self.stopped.wait(self.t1):
			self.update_models()

	def write_models(self):
		pickle.dump(self.models, open("modeldump.p","wb"))

	def update_models(self):
		curr_time = datetime.utcnow()
		buffer_copy = dict(self.handler.get_buffer())
		for loc,tweets in buffer_copy.items():
			if loc not in self.models:
				self.models[loc] = MLEModel(loc)
			self.models[loc].update_priors(tweets,int(curr_time.timestamp())-self.t0)
			self.models[loc].update_posteriors(tweets,int(curr_time.timestamp())-self.t1)
			self.models[loc].update_kld_distribution()
		
		self.write_models()
		print("Models updated.")


		potential_anomalies = {}
		for (loc,model) in self.models.items():
			# if divergence of tweets is 1 std away from mean, there are potential anomalies
			if abs(model.kl_divergence()-model.get_mean_kld()) > model.get_dev_kld():
				if loc in buffer_copy:
					potential_anomalies[loc] = buffer_copy[loc] # tweets that caused the anomaly by location
		self.handler.clear_buffer()

		if len(potential_anomalies) > 0:
			print("Potential anomalies found.")
			pickle.dump(potential_anomalies, open("potential_anomalies.p","wb"))

