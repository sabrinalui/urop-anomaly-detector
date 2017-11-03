from collections import defaultdict
import numpy as np

class MLEModel:

    def __init__(self, location):
        # location for now is represented by (lang,timezone) and is unique to the model
        self.location = location
        # initialize prior and posterior dicts, where key=term and value=ascending list of timestamps at which the term was tweeted
        self.priors = {}
        self.posteriors = {}
        self.priors_size = 0
        self.posteriors_size = 0

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
            new_start = np.argmax(np.array(v)>t0)
            self.priors[k] = [new_start:]
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
            new_start = np.argmax(np.array(v)>t0)
            self.posteriors[k] = [new_start:]
            self.posteriors_size -= new_start

        ### ADD NEW TERMS ####
        for (body,t) in tweets:
            for w in body.split():
                self.posteriors_size += 1
                self.update_dic(w,t,self.posteriors)

    def get_prior_mle(self,alpha):
        distribution = {}
        V = len(self.priors.keys())
        for (term,ts) in self.priors.items():
            distribution[term] = (len(ts) + alpha)/(self.priors_size + alpha*V)

        return distribution



