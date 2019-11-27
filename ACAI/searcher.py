import numpy as np
from numpy import argmax
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from warnings import catch_warnings
from warnings import simplefilter


class Searcher:
    def __init__(self, graph, search_method='Bayesian'):
        # Only support Bayesian optimization for now
        assert search_method == 'Bayesian'
        self.graph = graph
        self.log = None
        self.samples = None
        self.model = GaussianProcessRegressor()
        self.build_samples()

    def build_samples(self):
        samples = [[]]
        for node in self.graph:
            for hp in node.hyper_parameter:
                assert hp['type'] == 'float'
                if hp['type'] != 'float':
                    continue
                new_samples = []
                for cur_combo in samples:
                    count = int((hp['end'] - hp['start']) / hp['step_size'] + 1)
                    value = hp['start']
                    while count > 0:
                        new_hp = cur_combo.copy()
                        new_hp.append(value)
                        new_samples.append(new_hp)
                        value += hp['step_size']
                        count -= 1
                samples = new_samples
        self.samples = samples

    def get_next_hps(self, last_hps, last_rst):
        if not self.log:
            self.log = ([], [])
            # Random select a hp combo for the 1st try
            ind = np.random.randint(len(self.samples))
        else:
            # Update log
            last_sample = self.dict2list(last_hps)
            self.log[0].append(last_sample)
            self.log[1].append([last_rst])
            # calculate the best surrogate score found so far
            yhat, _ = self.surrogate(self.log[0])
            best = max(yhat)
            # calculate mean and stdev via surrogate function
            mu, std = self.surrogate(self.samples)
            mu = mu[:, 0]
            # calculate the probability of improvement
            scores = norm.cdf((mu - best) / (std+1E-9))
            # locate the index of the largest scores
            ind = argmax(scores)
        sample = self.samples[ind]
        del self.samples[ind]
        return self.list2dict(sample)

    def list2dict(self, sample):
        hps = {}
        ind = 0
        for node in self.graph:
            node_name = node.node_name
            hps[node_name] = {}
            for hp in node.hyper_parameter:
                hps[node_name][hp['name']] = sample[ind]
                ind += 1
        return hps

    def dict2list(self, hps):
        sample = []
        for node in self.graph:
            for hp in node.hyper_parameter:
                sample.append(hps[node.node_name][hp['name']])
        return sample

    def surrogate(self, samples):
        # catch any warning generated when making a prediction
        with catch_warnings():
            # ignore generated warnings
            simplefilter("ignore")
            return self.model.predict(samples, return_std=True)