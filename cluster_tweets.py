from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.externals import joblib

import pickle
import ipdb
import re
import numpy as np
import matplotlib.pyplot as plt
from doc2vectransform import doc2vectorizer

from pyclustering.cluster.xmeans import xmeans, splitting_type
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils import timedcall, euclidean_distance_square;


def run_kmeans(loc,tweets):
    vectorizer = CountVectorizer(stop_words=['rt', 'url'])
    Y = vectorizer.fit_transform(tweets)
    km = KMeans(n_clusters=60)
    km.fit(Y)

    order_centroids = km.cluster_centers_.argsort()[:, ::-1] 
    for i in range(len(order_centroids)):
        print("Cluster %d words: " % i, end='')
        terms = [vectorizer.get_feature_names()[j] for j in order_centroids[i, :5]]
        print(", ".join(terms))

def run_xmeans(loc,tweets): 
    # vectorizer = TfidfVectorizer(stop_words=['rt','url'],use_idf=True, ngram_range=(1,3))
    vectorizer = CountVectorizer(stop_words=['rt', 'url'])
    Y = vectorizer.fit_transform(tweets).toarray()

    # Y = doc2vectorizer(tweets)
    # naive cluster number 
    # n = 60
    # km = KMeans(n_clusters=n)
    # km.fit(Y)

    criterion = splitting_type.BAYESIAN_INFORMATION_CRITERION
    kmax = 20
    tolerance = .025
    initial_centers = kmeans_plusplus_initializer(Y, min(5,len(Y))).initialize()

    xmeans_instance = xmeans(Y,initial_centers,kmax,tolerance,criterion,False)
    (ticks,_) = timedcall(xmeans_instance.process)

    clusters = xmeans_instance.get_clusters()
    centers = xmeans_instance.get_centers()

    print(loc, ":", "\nNumber of clusters:", len(clusters), "\nExecution time:", ticks)

    raw_clusters = []
    for cluster in clusters:
        raw_cluster = []
        for i in cluster:
            raw_cluster.append(Y[i])
        raw_clusters.append(raw_cluster)

    # pickle the clusters and centers 
    clusters_pkl = get_clusters_pkl_name(loc)
    centers_pkl = get_centers_pkl_name(loc)
    joblib.dump(raw_clusters, clusters_pkl)
    joblib.dump(centers, centers_pkl)

    get_clusters(loc,vectorizer)

def cluster_significance(center,points):
    sigma_sqrt = 0
    for t in points:
        sigma_sqrt += euclidean_distance_square(t,center)
    return round(sigma_sqrt,2)

def get_clusters(loc,vectorizer):
    print("Centroids for %s:" % loc)
    centers = joblib.load(get_centers_pkl_name(loc))
    clusters = joblib.load(get_clusters_pkl_name(loc))

    index_centers = np.array(centers).argsort()[:, ::-1]
    for i in range(len(centers)):
        print("Cluster", i, "(", cluster_significance(centers[i],clusters[i]),"):",end=' ')
        terms = [vectorizer.get_feature_names()[j] for j in index_centers[i, :5]]
        print(", ".join(terms))

def get_pkl_name(loc):
    tz = re.split('[^a-zA-Z]',loc)[0]
    return 'pkl/' + tz + '.p'

def get_clusters_pkl_name(loc):
    tz = re.split('[^a-zA-Z]',loc)[0]
    return 'pkl/' + tz + '_clusters.p'

def get_centers_pkl_name(loc):
    tz = re.split('[^a-zA-Z]',loc)[0]
    return 'pkl/' + tz + '_centers.p'

if __name__=='__main__':
    with open('potential_anomalies.p', 'rb') as f:
        potential_anomalies = pickle.load(f)

    for (k,v) in potential_anomalies.items():
        tweet_bodies = [i[0] for i in v]
        run_xmeans(k,tweet_bodies)