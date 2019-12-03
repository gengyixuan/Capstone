import json
import os
import nltk
import torch

from torchtext import data
from torchtext import datasets
import pickle as pkl

def word_tokenize(tokens):
    return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]


class SQuAD():
    def __init__(self, hps):
        path = 'data/squad'
        dataset_path = path + '/torchtext/'

        print("preprocessing data files...")
        # self.preprocess_file('data/squad/train-v1.1.json', "data/train")
        # self.preprocess_file('data/squad/dev-v1.1.json', "data/dev")

        self.RAW = data.RawField()
        # explicit declaration for torchtext compatibility
        self.RAW.is_target = False
        self.CHAR_NESTING = data.Field(batch_first=True, tokenize=list, lower=True)
        self.CHAR = data.NestedField(self.CHAR_NESTING, tokenize=word_tokenize)
        self.WORD = data.Field(batch_first=True, tokenize=word_tokenize, lower=True, include_lengths=True)
        self.LABEL = data.Field(sequential=False, unk_token=None, use_vocab=False)

        dict_fields = {'id': ('id', self.RAW),
                       's_idx': ('s_idx', self.LABEL),
                       'e_idx': ('e_idx', self.LABEL),
                       'context': [('c_word', self.WORD), ('c_char', self.CHAR)],
                       'question': [('q_word', self.WORD), ('q_char', self.CHAR)]}

        list_fields = [('id', self.RAW), ('s_idx', self.LABEL), ('e_idx', self.LABEL),
                       ('c_word', self.WORD), ('c_char', self.CHAR),
                       ('q_word', self.WORD), ('q_char', self.CHAR)]


        print("building splits...")
        self.train, self.dev = data.TabularDataset.splits(
            path="./",
            train="data/train",
            validation="data/dev",
            format='json',
            fields=dict_fields)
        #print([e.c_word for e in self.train.examples])

        #cut too long context in the training set for efficiency.
        if hps["context_threshold"] > 0:
            self.train.examples = [e for e in self.train.examples if len(e.c_word) <= hps["context_threshold"]]

        print("building vocab...")
        self.CHAR.build_vocab(self.train, self.dev)
        self.WORD.build_vocab(self.train, self.dev)

        print("building iterators...")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(hps['train_batch_size'], hps['dev_batch_size'])
        self.train_iter, self.dev_iter = \
            data.BucketIterator.splits((self.train, self.dev),
                                       batch_sizes=[hps["train_batch_size"], hps["dev_batch_size"]],
                                       device=device,
                                       sort_key=lambda x: len(x.c_word))
        print(len(self.train_iter), len(self.dev_iter))

    def preprocess_file(self, path, outname):
        dump = []
        abnormals = [' ', '\n', '\u3000', '\u202f', '\u2009']

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            data = data['data']

            for article in data:
                for paragraph in article['paragraphs']:
                    context = paragraph['context']
                    tokens = word_tokenize(context)
                    for qa in paragraph['qas']:
                        id = qa['id']
                        question = qa['question']
                        for ans in qa['answers']:
                            answer = ans['text']
                            s_idx = ans['answer_start']
                            e_idx = s_idx + len(answer)

                            l = 0
                            s_found = False
                            for i, t in enumerate(tokens):
                                while l < len(context):
                                    if context[l] in abnormals:
                                        l += 1
                                    else:
                                        break
                                # exceptional cases
                                if t[0] == '"' and context[l:l + 2] == '\'\'':
                                    t = '\'\'' + t[1:]
                                elif t == '"' and context[l:l + 2] == '\'\'':
                                    t = '\'\''

                                l += len(t)
                                if l > s_idx and s_found == False:
                                    s_idx = i
                                    s_found = True
                                if l >= e_idx:
                                    e_idx = i
                                    break

                            dump.append(dict([('id', id),
                                              ('context', context),
                                              ('question', question),
                                              ('answer', answer),
                                              ('s_idx', s_idx),
                                              ('e_idx', e_idx)]))
        dump = dump[:1000]
        with open(outname, 'w', encoding='utf-8') as f:
            for line in dump:
                json.dump(line, f)
                print('', file=f)

# hps: context_threshold, word_dim, train_batch_size, dev_batch_size
def preprocess(inputs, hps):
    hps['word_dim'] = 50 # using pretrained Glove
    data = SQuAD(hps)
    iterlist = [(b.c_word, b.c_char, b.q_word, b.q_char, b.s_idx, b.e_idx, b.id) for b in data.dev_iter]
    training = list(iterlist[:200])
    development = list(iterlist[200:])
    char_vocab_len = len(data.CHAR.vocab)
    word_vocab_len = len(data.WORD.vocab)
    data_word_vocab_itos = data.WORD.vocab.itos

    rst = training, development, char_vocab_len, word_vocab_len, data_word_vocab_itos

    return rst
