import glog as log
from boom.modules import Module
from boom.log import set_logger

from collections import defaultdict
import time
import random
import torch
import pickle

class Preprocess(Module):

    def __init__(self, module_id, name, exp_name, rabbitmq_host, pipeline_conf, module_conf, **kwargs):
        super(Preprocess, self).__init__(module_id, name, exp_name, rabbitmq_host, pipeline_conf, module_conf, **kwargs)

        # Initialize logger.
        set_logger(rabbitmq_host, exp_name)


    def process(self, job, data):

        log.debug(job)
        # datalist = data

        # Functions to read in the corpus
        w2i = defaultdict(lambda: len(w2i))
        t2i = defaultdict(lambda: len(t2i))
        UNK = w2i["<unk>"]

        training_data = data["train"]
        validation_data = data["valid"]

        def read_dataset(dataset):
            for datapoint in dataset:
                tag, words, numtag = datapoint["tag"], datapoint["words"], len(datapoint["tag"])
                yield ([w2i[x] for x in words.split(" ")], t2i[tag], numtag)

        # Read in the data
        basket = {}
        basket['train'] = list(read_dataset(training_data))
        basket['w2i'] = defaultdict(lambda: UNK, w2i)
        basket['t2i'] = t2i
        basket['dev'] = list(read_dataset(validation_data))
        basket['nwords'] = len(w2i)
        basket['ntags'] = len(t2i)


        return basket