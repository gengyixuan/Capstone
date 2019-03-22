from cz_classifier_word2vec import classify_word2vec
from utils import print_metrics


def main():
    # Set hyper parameters
    dimension = 300
    shuffle = False
    training_ratio = 0.8
    metrics_path = "output/metrics_word2vec.csv"

    # Evaluate multiple models
    rst = dict()
    rst.update(classify_word2vec(dimension, shuffle, training_ratio))
    # TODO: other models (rst.update(balabala))

    # print metrics
    print_metrics(rst, metrics_path)


if __name__ == '__main__':
    main()
