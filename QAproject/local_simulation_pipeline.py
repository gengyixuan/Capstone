from __future__ import print_function
import train
import evaluate
import preprocess

import json
import os
import nltk
import torch

from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe


from collections import Counter
import string
import re
import argparse
import json
import sys

import argparse
import copy, json, os

import torch
from torch import nn, optim
from time import gmtime, strftime

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.nn import LSTM, Linear

from train import *
from evaluate import *
from preprocess import *

import pickle as pkl

hps = {
    "train_batch_size": 4,
    "dev_batch_size": 4,
    "context_threshold": 10,
    "word_dim": 50
}
out1 = preprocess(None, hps)

hps = {
    "exp_decay_rate": 0.1,
    "learning_rate": 0.001,
    "epoch": 1,
    "char_dim": 8,
    "char_channel_width": 4, # smaller than 4 please
    "hidden_size": 64, # must be higher than 50
    "dropout": 0.8,
}
inputs = {
    "preprocess": out1
}
out2 = train(inputs, hps)
pkl.dump(out2, open('train.pkl', 'wb'))
in3 = pkl.load(open('train.pkl', 'rb'))

inputs = {
    "preprocess": out1,
    "train": in3
}
out3 = evaluate(inputs, None)

print(out3)