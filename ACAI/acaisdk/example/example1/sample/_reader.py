from src.reader import reader
import argparse
import pickle as pkl
import os
parser=argparse.ArgumentParser()
parser.add_argument('--mult', type=float)
parser.add_argument('--add', type=float)
args=parser.parse_args()
inputs=dict()
hps=dict()
hps['mult']=args.mult
hps['add']=args.add
rst=reader(inputs, hps)
os.mkdir('reader_output')
pkl.dump(rst, open('reader_output/reader.pkl', 'wb'))
