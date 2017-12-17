import ipdb
import pickle
import os

with open("modeldump.p", "rb") as f:
	models = pickle.load(f)
with open("potential_anomalies.p", "rb") as f:
	anomalies = pickle.load(f)
ipdb.set_trace()