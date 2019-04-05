import json
from utils import save_metrics
from data_model_list import data_model_list
from classifier import model_training


config_path = "config.json"
with open(config_path, 'r') as config_file:
    config = json.load(config_file)

metrics = config['metrics'].split(',')
data_models = config['data_models'].split(',')
classifiers = config['classifiers'].split(',')
# metrics = ['Precision', 'Recall', 'F1']
# data_models = ['WordEmbedding', 'Metadata', 'Combined']
# classifiers = ['SVM', 'NaiveBayes', 'MLP']
raw_text_path = config['raw_data_path']
test_ratio = 1.0 - config['train_ratio']
metrics_path = config['metrics_path']
verbose_mode = config['verbose_mode']


def iterate_through_classifiers(X, Y):
    rst_classifiers = dict()
    for classifier in classifiers:
        report = model_training(X, Y, test_ratio, verbose_mode, classifier)
        rst_classifiers.update({classifier: report})
    return rst_classifiers


def iterate_through_data_models():
    rst_data_models = dict()
    for data_model in data_models:
        X, Y = data_model_list[data_model](raw_text_path, verbose_mode)
        report = iterate_through_classifiers(X, Y)
        rst_data_models.update({data_model: report})
    return rst_data_models


def main():
    rst = iterate_through_data_models()
    # save metrics
    save_metrics(rst, metrics_path, metrics, data_models, classifiers)


if __name__ == '__main__':
    main()
