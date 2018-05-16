import tweepy
import yaml
from stream_handler import *
from oauth import CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET
from threading import Timer,Thread,Event
import sys
import time
from cluster_tweets import *
from model import MLEModel
import matplotlib.pyplot as plt

# English stopwords taken from NLTK
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']
sqlite_file = 'tweets.db'

def create_table():
	# create sqlite3 tweets table
	create_table_sql = """ CREATE TABLE IF NOT EXISTS tweets (
								loc text NOT NULL,
								body text NOT NULL,
								time_stamp integer NOT NULL
						); """

	print("Creating SQLite table...")
	conn = sqlite3.connect(sqlite_file)
	try:
		c = conn.cursor()
		c.execute(create_table_sql)
	except sqlite3.Error as e:
		print("An error occurred:", e.args[0])
	conn.commit()
	conn.close()
	print("Table created.")

def clear_table():
	print("Clearing SQLite tweets table...")
	conn = sqlite3.connect(sqlite_file)
	try: 
		c = conn.cursor()
		c.execute('delete from tweets')
	except sqlite3.Error as e:
		print("An error occurred:", e.args[0])
	conn.commit()
	conn.close()

def gen_models(file):
	conn = sqlite3.connect(sqlite_file)
	counts = []
	try: 
		c = conn.cursor()
		c.execute('select loc, count(1) from tweets group by loc')
		counts = c.fetchall()
	except sqlite3.Error as e:
		print("An error occurred:", e.args[0])
	conn.commit()
	conn.close()

	cities = [i[0] for i in counts if i[1] > 50]
	T0, T1, T2 = (1526158083,1526233869,1526237469)

	# Generate models for cities with significant data
	models = {}
	potential_anomalies = {}
	conn = sqlite3.connect(sqlite_file)
	c = conn.cursor()
	for city in cities:
		c.execute('select * from tweets where loc = ? and time_stamp between ? and ?',[city,T0,T1])
		prior = c.fetchall()
		prior = [(i[1],i[2]) for i in prior]
		c.execute('select * from tweets where loc = ? and time_stamp between ? and ?',[city,T1,T2])
		posterior = c.fetchall()
		posterior = [(i[1],i[2]) for i in posterior]
		if len(prior) > 0 and len(posterior) > 0:
			print("Creating", city, "model")
			potential_anomalies[city] = posterior
			models[city] = MLEModel(city)
			models[city].update_priors(prior,0)
			models[city].update_posteriors(posterior,0)
			models[city].update_kld_distribution()
	conn.commit()
	conn.close()

	pickle.dump(models, open("modeldump.p","wb"))
	pickle.dump(potential_anomalies, open("potential_anomalies.p","wb"))
	print("Models saved")

if __name__=='__main__':

	# initialize params in seconds
	t0 = 86400 # 24h
	t1 = 3600 # 1h
	if len(sys.argv) == 3:
		t0 = int(sys.argv[1])
		t1 = int(sys.argv[2])

	listener = StreamingAnomalyHandler()
	updater = UpdaterThread(Event(),t0,t1,listener)
	updater.start()

	auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
	auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

	api = tweepy.API(auth)

	twitterstream = tweepy.Stream(auth=api.auth, listener=listener)
	twitterstream.filter(track=stop_words,async=True)

	time.sleep(t0)
	twitterstream.disconnect()
	# thread is guaranteed to EVENTUALLY stop running since tweets are no longer being written to the table
