from utils import print_metrics
from data_model_list import data_model_list
from classifier import model_training

data_models = ['WordEmbedding', 'Metadata', 'Combined']
classifiers = ['SVM', 'NaiveBayes', 'MLP']
raw_text_path = "topicclass/topicclass_train_small.txt"
verbose_mode = False
test_ratio = 0.2
metrics_path = "output/metrics_word2vec.csv"
rst = dict()


def iterate_through_classifiers(X, Y):
    for classifier in classifiers:
        report = model_training(X, Y, test_ratio, verbose_mode, classifier)
        rst.update(report)


def iterate_through_data_models():
    for name in data_models:
        X, Y = data_model_list[name](raw_text_path, verbose_mode)
        iterate_through_classifiers(X, Y)


def main():
    iterate_through_data_models()
    print(rst)
    # print metrics
    print_metrics(rst, metrics_path)


if __name__ == '__main__':
    main()
