from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
import pickle
import ipdb
import re

def run_clustering(loc,tweets): 
    vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1,3))
    X = vectorizer.fit_transform(tweets)

    # naive cluster number 
    n = max(int(len(tweets)/4),1)
    if n > len(tweets):
        n = len(tweets)
    km = KMeans(n_clusters=n)
    km.fit(X)

    # dumps model to pickle file
    pkl = get_pkl_name(loc)
    joblib.dump(km, pkl)

    get_clusters(loc,vectorizer)

def get_clusters(loc,vectorizer):
    print("CLUSTERING for %s:" % loc)
    pkl = get_pkl_name(loc)
    km = joblib.load(pkl)
    num_clusters = len(km.cluster_centers_)
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    for i in range(num_clusters):
        print("Cluster %d words: " % i, end='')
        terms = [vectorizer.get_feature_names()[j] for j in order_centroids[i, :5]]
        print(",".join(terms))

def get_pkl_name(loc):
    tz = re.split('[^a-zA-Z]',loc)[0]
    return 'pkl/' + tz + '.p'

if __name__=='__main__':
    with open('potential_anomalies.p', 'rb') as f:
        potential_anomalies = pickle.load(f)
    # test_loc = ('en', 'America/New_York')
    # if test_loc in potential_anomalies:
    #     tweet_bodies = [i[0] for i in potential_anomalies[test_loc]]
    #     run_clustering(test_loc,tweet_bodies)
    for (k,v) in potential_anomalies.items():
        tweet_bodies = [i[0] for i in v]
        run_clustering(k,tweet_bodies)