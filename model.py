from collections import defaultdict
import numpy as np
import math
from sklearn.feature_extraction.text import CountVectorizer

class MLEModel:

    def __init__(self, location):

        # location for now is represented by (lang,timezone) and is unique to the model
        self.location = location

        # initialize prior and posterior dicts, where key=term and value=ascending list of timestamps at which the term was tweeted
        self.priors = {}
        self.posteriors = {}
        self.priors_size = 0
        self.posteriors_size = 0

        # EWMA mean and std dev of KL divergence
        self.mean_kld = None 
        self.dev_kld = None

    # update unigram count for one term
    def update_dic(self,w,t,dic):
        if w in dic:
            dic[w].append(t)
        else:
            dic[w] = [t]

    def update_priors(self,tweets,t0):
        # tweets are sanitized and in the tuple form (body, timestamp)

        #### GET RID OF OLD TERMS ####
        # for each term in priors, shift window by slicing out timestamps older than t0
        for (k,v) in self.priors.items():
            if len(v) > 0:
                if (np.array(v)>t0).all() == False: # all tweets are stale
                    self.priors_size -= len(v)
                    self.priors[k] = []
                else:
                    new_start = np.argmax(np.array(v)>t0)
                    self.priors[k] = self.priors[k][new_start:]
                    self.priors_size -= new_start

        ### ADD NEW TERMS ####
        for (body,t) in tweets:
            for w in body.split():
                self.priors_size += 1
                self.update_dic(w,t,self.priors)

    def update_posteriors(self,tweets,t0):
        # tweets are sanitized and in the tuple form (body, timestamp)

        #### GET RID OF OLD TERMS ####
        # for each term in priors, shift window by slicing out timestamps older than t0
        for (k,v) in self.posteriors.items():
            if len(v) > 0:
                if (np.array(v)>t0).all() == False: # all tweets are stale
                    self.posteriors_size -= len(v)
                    self.posteriors[k] = []
                else:
                    new_start = np.argmax(np.array(v)>t0)
                    self.posteriors[k] = self.posteriors[k][new_start:]
                    self.posteriors_size -= new_start

        ### ADD NEW TERMS ####
        for (body,t) in tweets:
            for w in body.split():
                self.posteriors_size += 1
                self.update_dic(w,t,self.posteriors)

    # For now, prior and posterior are calculated independent of each other using MLE with add-alpha discounting
    
    def get_prior_mle(self,alpha):
        distribution = {}
        V = len(self.priors.keys())
        for (term,ts) in self.priors.items():
            distribution[term] = (len(ts) + alpha)/(self.priors_size + alpha*V)

        alpha_boost = alpha / (self.priors_size + alpha*V)
        return defaultdict(lambda: alpha_boost, distribution)

    def get_posterior_mle(self,alpha):
        distribution = {}
        V = len(self.posteriors.keys())
        for (term,ts) in self.posteriors.items():
            distribution[term] = (len(ts) + alpha)/(self.posteriors_size + alpha*V)

        alpha_boost = alpha / (self.posteriors_size + alpha*V)
        return defaultdict(lambda: alpha_boost, distribution)

    # Returns D_KL, divergence from prior to posterior 
    def kl_divergence(self):
        alpha = 0.01
        d_kl = 0
        prior_mle = self.get_prior_mle(alpha)
        post_mle = self.get_posterior_mle(alpha)
        for (word,prob) in list(post_mle.items()):
            d_kl += prob * math.log(prob/(prior_mle[word]))
        return d_kl

    def get_mean_kld(self):
        return self.mean_kld

    def get_dev_kld(self):
        return self.dev_kld

    def update_kld_distribution(self):
        alpha = 0.125
        beta = 0.25

        r = self.kl_divergence()
        if self.mean_kld == None:
            self.mean_kld = r
        else:
            self.mean_kld = alpha*r + (1-alpha)*self.mean_kld
        dev = abs(r-self.mean_kld)
        if self.dev_kld == None:
            self.dev_kld = dev
        else:
            self.dev_kld = beta*dev + (1-beta)*self.dev_kld

    def __repr__(self):
        return "KLD: " + str(round(self.kl_divergence(),4)) + ", mean: " + str(round(self.get_mean_kld(),4)) + ", std dev: " + str(round(self.get_dev_kld(),4))



