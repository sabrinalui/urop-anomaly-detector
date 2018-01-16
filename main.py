import tweepy
import yaml
from stream_handler import *
from oauth import CONSUMER_KEY, CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET
from threading import Timer,Thread,Event
import sys
import time

# English stopwords taken from NLTK
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']

def create_table():
	# create sqlite3 tweets table
	create_table_sql = """ CREATE TABLE IF NOT EXISTS tweets (
								id integer NOT NULL,
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

if __name__=='__main__':

	# initialize params in seconds
	t0 = 7200
	t1 = 1800
	if len(sys.argv) == 3:
		t0 = int(sys.argv[1])
		t1 = int(sys.argv[2])

	clear_table()
	print("Cleared table")

	runtime = 10

	listener = StreamingAnomalyHandler()
	updater = UpdaterThread(Event(),t0,t1,listener)
	updater.start()

	auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
	auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

	api = tweepy.API(auth)

	twitterstream = tweepy.Stream(auth=api.auth, listener=listener)
	twitterstream.filter(track=stop_words,async=True)

	time.sleep(runtime)
	twitterstream.disconnect()
	# thread is guaranteed to EVENTUALLY stop running since tweets are no longer being written to the table
