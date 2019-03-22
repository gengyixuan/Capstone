from cz_classifier_word2vec import classify_word2vec
from yf_model_training import classify_concat
from yf_pre_processing import concat_preprocess
from gyx_meta_maker import meta_preprocess
from gyx_classifier import classify_meta
from utils import print_metrics
from utils import report_parser


def main():
    # Set hyper parameters
    dimension = 300
    shuffle = False
    training_ratio = 0.8
    verbose_mode = False
    metrics_path = "output/metrics_word2vec.csv"
    raw_text_path = "topicclass/topicclass_train_small.txt"

    rst = dict()

    # Evaluate multiple models
    print("word2vec training")
    ret_report_w2v = classify_word2vec(dimension, shuffle, training_ratio, raw_text_path, verbose_mode)
    rst.update(ret_report_w2v)

    print("meta training")
    meta_preprocess(raw_text_path, verbose_mode)
    ret_report_meta = classify_meta(raw_text_path, training_ratio, verbose_mode)
    rst.update(ret_report_meta)

    print("concat training")
    concat_preprocess(raw_text_path, verbose_mode)
    ret_report_concat = classify_concat(training_ratio, verbose_mode)
    rst.update(ret_report_concat)

    print(rst)

    # print metrics
    # print_metrics(rst, metrics_path)


if __name__ == '__main__':
    main()
