""" Customized version of the official evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys
import torch
from torch import nn


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def eval(dataset, predictions):
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    #print(message, file=sys.stderr)
                    continue
                total += 1
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1} 

def test(model, deviter, data_word_vocab_itos):
    print("start test")
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    loss = 0
    answers = dict()
    model.eval()

    with torch.set_grad_enabled(False):
        for batch in deviter:
            p1, p2 = model(batch)
            b_c_word, b_c_char, b_q_word, b_q_char, b_s_idx, b_e_idx, b_id = batch
            batch_loss = criterion(p1, b_s_idx) + criterion(p2, b_e_idx)
            loss += batch_loss.item()

            # (batch, c_len, c_len)
            batch_size, c_len = p1.size()
            ls = nn.LogSoftmax(dim=1)
            mask = (torch.ones(c_len, c_len) * float('-inf')).to(device).tril(-1).unsqueeze(0).expand(batch_size, -1, -1)
            score = (ls(p1).unsqueeze(2) + ls(p2).unsqueeze(1)) + mask
            score, s_idx = score.max(dim=1)
            score, e_idx = score.max(dim=1)
            s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze()

            for i in range(batch_size):
                id = b_id[i]
                answer = b_c_word[0][i][s_idx[i]:e_idx[i]+1]
                answer = ' '.join([data_word_vocab_itos[idx] for idx in answer])
                answers[id] = answer

    predictions = answers
    with open("data/squad/dev-v1.1.json") as dataset_file:
        dataset_json = json.load(dataset_file)
        """
        if (dataset_json['version'] != expected_version):
            print('Evaluation expects v-' + expected_version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        """
        dataset = dataset_json['data']

    results = eval(dataset, predictions)
    results['loss'] = loss
    return results #results['loss'], results['exact_match'], results['f1']

def evaluate(inputs, hps):
    trainiter, deviter, char_vocab_len, word_vocab_len, data_word_vocab_itos = inputs['preprocess']
    return test(inputs['train'], deviter, data_word_vocab_itos)