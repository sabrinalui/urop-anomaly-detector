import tweepy
import sqlite3
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

sqlite_file = 'tweets.db'
insert_row_sql = """ INSERT INTO tweets(id,loc,body,time_stamp)
              VALUES(?,?,?,?) """
query_rows_sql = "select * from tweets where time_stamp between ? and ?"
delete_rows_sql = "delete from tweets where time_stamp < ?"

class StreamingAnomalyHandler(tweepy.StreamListener):

	def __init__(self):
		self.tweet_buffer = []

		super(StreamingAnomalyHandler, self).__init__()

	def on_status(self, status):
		tweet_id = status.id
		if status.user.lang == "en": # restrict to English for now
			(tweet,loc) = self.sanitize(status)
			body = tweet[0]
			timestamp = tweet[1]
			if loc != None: # insert into sqlite table
				tweet_tup = (tweet_id,loc,body,timestamp)
				# self.tweet_buffer.append(tweet_tup)
				conn = sqlite3.connect(sqlite_file)
				try:
					c = conn.cursor()
					c.execute(insert_row_sql,tweet_tup)
					# print("Added row " + str(c.lastrowid))
				except sqlite3.Error as e:
					print("An error occurred:", e.args[0])
				conn.commit()
				conn.close()

	def sanitize(self, status):
		body_words = status.text.split()
		timestamp = int(status.created_at.astimezone(pytz.utc).timestamp())
		filtered_words = [word for word in body_words if word.lower() not in stop_words]
		body = ' '.join(filtered_words)
		body = re.sub(r"http\S+", "<URL>",body)
		# loc = (status.user.time_zone, status.user.lang)
		loc = status.user.time_zone
		return ((body,timestamp),loc)

	def write_tweets_and_clear_buffer(self):
		# write tweets in buffer to sqlite table and clear buffer
		write_conn = sqlite3.connect(sqlite_file)
		try:
			c = write_conn.cursor()
			for tweet in self.tweet_buffer:
				c.execute(insert_row_sql,tweet)
				print("Added row", c.lastrowid)
		except sqlite3.Error as e:
			print("An error occurred:", e.args[0])
		write_conn.commit()
		write_conn.close()
		self.tweet_buffer = []


class UpdaterThread(Thread):

	def __init__(self,event,t0,t1,handler):
		Thread.__init__(self)
		self.stopped = event
		self.models = {}
		self.t0, self.t1 = t0,t1
		self.handler = handler

	def run(self):
		while not self.stopped.wait(self.t1):
			running = self.update_models()
			if not running:
				raise ValueError("Stream stopped so updater stopped")

	def write_models(self):
		pickle.dump(self.models, open("modeldump.p","wb"))

	def update_models(self):
		curr_time = int(datetime.utcnow().timestamp())
		cutoff_time = curr_time-self.t1
		queue = defaultdict(list)

		# fetch tweets from the past t1 seconds and delete from table
		tweets = []
		read_and_delete_conn = sqlite3.connect(sqlite_file)
		try:
			c = read_and_delete_conn.cursor()
			c.execute(query_rows_sql,[cutoff_time,curr_time])
			tweets = c.fetchall()
			print("Fetched", len(tweets), "tweets")
			del_ct = c.execute(delete_rows_sql,(cutoff_time,)).rowcount
			print("Deleted", del_ct, "rows")
		except sqlite3.Error as e:
			print("An error occurred:", e.args[0])
		read_and_delete_conn.commit()
		read_and_delete_conn.close()

		# abort thread if stream has stopped writing to table
		if (len(tweets) == 0):
			return False

		# otherwise update model normally
		for tweet in tweets:
			queue[tweet[1]].append((tweet[2],tweet[3]))

		for loc,tweets in queue.items():
			if loc not in self.models:
				self.models[loc] = MLEModel(loc)
			self.models[loc].update_priors(tweets,curr_time-self.t0)
			self.models[loc].update_posteriors(tweets,curr_time-self.t1)
			self.models[loc].update_kld_distribution()
		
		self.write_models()
		print("Models updated")

		# potential_anomalies = {}
		# for (loc,model) in self.models.items():
		# 	# if divergence of tweets is 1 std away from mean, there are potential anomalies
		# 	if abs(model.kl_divergence()-model.get_mean_kld()) > model.get_dev_kld():
		# 		if loc in buffer_copy:
		# 			potential_anomalies[loc] = buffer_copy[loc] # tweets that caused the anomaly by location
		self.handler.write_tweets_and_clear_buffer()

		# if len(potential_anomalies) > 0:
		# 	print("Potential anomalies found.")
		# 	pickle.dump(potential_anomalies, open("potential_anomalies.p","wb"))

		return True

