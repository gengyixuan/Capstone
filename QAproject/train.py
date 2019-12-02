import argparse
import copy, json, os

import torch
from torch import nn, optim
from time import gmtime, strftime

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.nn import LSTM, Linear



# hps: 
class BiDAF(nn.Module):
    def __init__(self, hps):
        super(BiDAF, self).__init__()
        self.hps= hps
        # 1. Character Embedding Layer
        self.char_emb = nn.Embedding(hps["char_vocab_size"], hps["char_dim"], padding_idx=1)
        nn.init.uniform_(self.char_emb.weight, -0.001, 0.001)
        hps['char_channel_size'] = hps["hidden_size"] * 2 - hps["word_dim"]
        assert hps['char_channel_size'] > 0

        self.char_conv = nn.Conv2d(1, hps["char_channel_size"], (hps["char_dim"], hps["char_channel_width"]))

        # 2. Word Embedding Layer
        # initialize word embedding with GloVe
        self.word_emb = nn.Embedding(50000, 50)

        # highway network
        # assert self.hps["hidden_size"] * 2 == (self.hps["char_channel_size"] + self.hps["word_dim"])
        for i in range(2):
            setattr(self, f'highway_linear{i}',
                    nn.Sequential(Linear(hps["hidden_size"] * 2, hps["hidden_size"] * 2),
                                  nn.ReLU()))
            setattr(self, f'highway_gate{i}',
                    nn.Sequential(Linear(hps["hidden_size"] * 2, hps["hidden_size"] * 2),
                                  nn.Sigmoid()))

        # 3. Contextual Embedding Layer
        self.context_LSTM = LSTM(input_size=hps["hidden_size"] * 2,
                                 hidden_size=hps["hidden_size"],
                                 bidirectional=True,
                                 batch_first=True,
                                 dropout=hps["dropout"])

        # 4. Attention Flow Layer
        self.att_weight_c = Linear(hps["hidden_size"] * 2, 1)
        self.att_weight_q = Linear(hps["hidden_size"] * 2, 1)
        self.att_weight_cq = Linear(hps["hidden_size"] * 2, 1)

        # 5. Modeling Layer
        self.modeling_LSTM1 = LSTM(input_size=hps["hidden_size"] * 8,
                                   hidden_size=hps["hidden_size"],
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=hps["dropout"])

        self.modeling_LSTM2 = LSTM(input_size=hps["hidden_size"] * 2,
                                   hidden_size=hps["hidden_size"],
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=hps["dropout"])

        # 6. Output Layer
        self.p1_weight_g = Linear(hps["hidden_size"] * 8, 1, dropout=hps["dropout"])
        self.p1_weight_m = Linear(hps["hidden_size"] * 2, 1, dropout=hps["dropout"])
        self.p2_weight_g = Linear(hps["hidden_size"] * 8, 1, dropout=hps["dropout"])
        self.p2_weight_m = Linear(hps["hidden_size"] * 2, 1, dropout=hps["dropout"])

        self.output_LSTM = LSTM(input_size=hps["hidden_size"] * 2,
                                hidden_size=hps["hidden_size"],
                                bidirectional=True,
                                batch_first=True,
                                dropout=hps["dropout"])

        self.dropout = nn.Dropout(p=hps["dropout"])

    def forward(self, batch):
        # TODO: More memory-efficient architecture
        b_c_word, b_c_char, b_q_word, b_q_char, b_s_idx, b_e_idx, b_id = batch
        def char_emb_layer(x):
            """
            :param x: (batch, seq_len, word_len)
            :return: (batch, seq_len, char_channel_size)
            """
            batch_size = x.size(0)
            # (batch, seq_len, word_len, char_dim)
            x = self.dropout(self.char_emb(x))
            # (batch * seq_len, 1, char_dim, word_len)
            x = x.view(-1, self.hps["char_dim"], x.size(2)).unsqueeze(1)
            # (batch * seq_len, char_channel_size, 1, conv_len) -> (batch * seq_len, char_channel_size, conv_len)
            x = self.char_conv(x).squeeze()
            # (batch * seq_len, char_channel_size, 1) -> (batch * seq_len, char_channel_size)
            x = F.max_pool1d(x, x.size(2)).squeeze()
            # (batch, seq_len, char_channel_size)
            x = x.view(batch_size, -1, self.hps["char_channel_size"])

            return x

        def highway_network(x1, x2):
            """
            :param x1: (batch, seq_len, char_channel_size)
            :param x2: (batch, seq_len, word_dim)
            :return: (batch, seq_len, hidden_size * 2)
            """
            # (batch, seq_len, char_channel_size + word_dim)
            x = torch.cat([x1, x2], dim=-1)
            print(x.shape, x1.shape, x2.shape)
            for i in range(2):
                h = getattr(self, f'highway_linear{i}')(x)
                g = getattr(self, f'highway_gate{i}')(x)
                x = g * h + (1 - g) * x
            # (batch, seq_len, hidden_size * 2)
            return x

        def att_flow_layer(c, q):
            """
            :param c: (batch, c_len, hidden_size * 2)
            :param q: (batch, q_len, hidden_size * 2)
            :return: (batch, c_len, q_len)
            """
            c_len = c.size(1)
            q_len = q.size(1)

            # (batch, c_len, q_len, hidden_size * 2)
            #c_tiled = c.unsqueeze(2).expand(-1, -1, q_len, -1)
            # (batch, c_len, q_len, hidden_size * 2)
            #q_tiled = q.unsqueeze(1).expand(-1, c_len, -1, -1)
            # (batch, c_len, q_len, hidden_size * 2)
            #cq_tiled = c_tiled * q_tiled
            #cq_tiled = c.unsqueeze(2).expand(-1, -1, q_len, -1) * q.unsqueeze(1).expand(-1, c_len, -1, -1)

            cq = []
            for i in range(q_len):
                #(batch, 1, hidden_size * 2)
                qi = q.select(1, i).unsqueeze(1)
                #(batch, c_len, 1)
                ci = self.att_weight_cq(c * qi).squeeze()
                cq.append(ci)
            # (batch, c_len, q_len)
            cq = torch.stack(cq, dim=-1)

            # (batch, c_len, q_len)
            s = self.att_weight_c(c).expand(-1, -1, q_len) + \
                self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + \
                cq

            # (batch, c_len, q_len)
            a = F.softmax(s, dim=2)
            # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)
            c2q_att = torch.bmm(a, q)
            # (batch, 1, c_len)
            b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
            # (batch, 1, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)
            q2c_att = torch.bmm(b, c).squeeze()
            # (batch, c_len, hidden_size * 2) (tiled)
            q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)
            # q2c_att = torch.stack([q2c_att] * c_len, dim=1)

            # (batch, c_len, hidden_size * 8)
            x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1)
            return x

        def output_layer(g, m, l):
            """
            :param g: (batch, c_len, hidden_size * 8)
            :param m: (batch, c_len ,hidden_size * 2)
            :return: p1: (batch, c_len), p2: (batch, c_len)
            """
            # (batch, c_len)
            p1 = (self.p1_weight_g(g) + self.p1_weight_m(m)).squeeze()
            # (batch, c_len, hidden_size * 2)
            m2 = self.output_LSTM((m, l))[0]
            # (batch, c_len)
            p2 = (self.p2_weight_g(g) + self.p2_weight_m(m2)).squeeze()

            return p1, p2

        # 1. Character Embedding Layer
        c_char = char_emb_layer(b_c_char)
        q_char = char_emb_layer(b_q_char)
        # 2. Word Embedding Layer
        c_word = self.word_emb(b_c_word[0])
        q_word = self.word_emb(b_q_word[0])
        c_lens = b_c_word[1]
        q_lens = b_q_word[1]

        # Highway network
        c = highway_network(c_char, c_word)
        q = highway_network(q_char, q_word)
        # 3. Contextual Embedding Layer
        c = self.context_LSTM((c, c_lens))[0]
        q = self.context_LSTM((q, q_lens))[0]
        # 4. Attention Flow Layer
        g = att_flow_layer(c, q)
        # 5. Modeling Layer
        m = self.modeling_LSTM2((self.modeling_LSTM1((g, c_lens))[0], c_lens))[0]
        # 6. Output Layer
        p1, p2 = output_layer(g, m, c_lens)

        # (batch, c_len), (batch, c_len)
        return p1, p2


class EMA():
    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def get(self, name):
        return self.shadow[name]

    def update(self, name, x):
        assert name in self.shadow
        new_average = (1.0 - self.mu) * x + self.mu * self.shadow[name]
        self.shadow[name] = new_average.clone()


# hps: exp_decay_rate, learning_rate, epoch
# char_vocab_size, char_dim, char_channel_width, hidden_size, word_dim, dropout
def train(inputs, hps):
    hps["char_dim"] = int(hps["char_dim"])
    hps["hidden_size"] = int(hps["hidden_size"])
    hps['char_channel_width'] = int(hps["char_channel_width"])
    hps['epoch'] = int(hps['epoch'])

    trainiter, deviter, char_vocab_len, word_vocab_len, data_word_vocab_itos = inputs['preprocess']
    hps['char_vocab_size'] = char_vocab_len
    hps['word_vocab_size'] = word_vocab_len
    hps['word_dim'] = 50  # fixed
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BiDAF(hps).to(device)

    ema = EMA(hps["exp_decay_rate"])
    for name, param in model.named_parameters():
        if param.requires_grad:
            ema.register(name, param.data)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters, lr=hps["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    # training

    model.train()
    loss, last_epoch = 0, -1
    max_dev_exact, max_dev_f1 = -1, -1

    iterator = trainiter
    for present_epoch in range(hps['epoch']):
        for i, batch in enumerate(iterator):
            b_c_word, b_c_char, b_q_word, b_q_char, b_s_idx, b_e_idx, b_id = batch
            last_epoch = present_epoch

            p1, p2 = model(batch)

            optimizer.zero_grad()
            batch_loss = criterion(p1, b_s_idx) + criterion(p2, b_e_idx)
            loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()

            for name, param in model.named_parameters():
                if param.requires_grad:
                    ema.update(name, param.data)

    return model