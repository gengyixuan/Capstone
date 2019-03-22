import numpy as np
from sklearn.pipeline import Pipeline
from sklearn import *
from collections import defaultdict
from utils import print_metrics
from utils import report_parser


# Build vectorizer
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec, dim):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = dim

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                     or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec, dim):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = dim

    def fit(self, X, y):
        tfidf = feature_extraction.text.TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])


def classify_word2vec(dimension, shuffle, training_ratio, raw_text_path, verbose_mode):

    word2vec_path = "word_embeddings/glove.6B.%dd.txt" % dimension
    # raw_text_path = "topicclass/topicclass_train_mid.txt"
    # Read pre-trained list
    with open(word2vec_path, "r") as lines:
        w2v = {line.split()[0]: np.array(list(map(float, line.split()[1:])))
               for line in lines}

    # Read from raw text
    label = []
    data = []
    with open(raw_text_path, "r") as lines:
        for line in lines:
            segs = line.split("|||")
            label.append(segs[0])
            data.append(segs[1].split())
    data = np.array(data)
    label = np.array(label)

    # Build pipeline
    models = dict()
    models['SVM'] = Pipeline([
        ('word2vec vectorizer', MeanEmbeddingVectorizer(w2v, dimension)),
        ('svm', svm.LinearSVC(random_state=0, tol=1e-5))
    ])
    models['Neural Network'] = Pipeline([
        ('word2vec vectorizer', MeanEmbeddingVectorizer(w2v, dimension)),
        ('neural network', neural_network.MLPClassifier(
                hidden_layer_sizes=(1000, ),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size='auto',
                learning_rate='constant',
                learning_rate_init=0.01,
                power_t=0.5,
                max_iter=10000,
                shuffle=True,
                random_state=None,
                tol=0.0001,
                verbose=verbose_mode,
                warm_start=False,
                momentum=0.9,
                nesterovs_momentum=True,
                early_stopping=True,
                validation_fraction=0.1,
                beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                n_iter_no_change=10))
    ])
    models['Extra Tree'] = Pipeline([
        ('word2vec vectorizer', MeanEmbeddingVectorizer(w2v, dimension)),
        ('extra tree', ensemble.ExtraTreesClassifier(n_estimators=200))
    ])

    # shuffle dataset
    if shuffle:
        data_label = np.c_[data.reshape(len(data), -1), label.reshape(len(label), -1)]
        np.random.shuffle(data_label)
        data = data_label[:, :data.size//len(data)].reshape(data.shape)
        label = data_label[:, data.size//len(data):].reshape(label.shape)

    # Split dataset
    train_data_size = int(len(data) * training_ratio)
    train_data = data[:train_data_size]
    train_label = label[:train_data_size]
    test_data = data[train_data_size:]
    test_label = label[train_data_size:]

    # Train and test
    rst = dict()
    for model in models:
        if verbose_mode: print("Start evaluating: %s" % model)
        rst[model] = dict()
        models[model].fit(train_data, train_label)
        test_pred = models[model].predict(test_data)
        entry = metrics.precision_recall_fscore_support(test_label, test_pred)
        accuracy, recall, f1, support = entry
        accuracy = np.mean(accuracy)
        recall = np.mean(recall)
        f1 = np.mean(f1)
        rst[model] = accuracy, recall, f1
        
        ret_report = metrics.classification_report(test_label, test_pred)
        rst[model] = report_parser(ret_report)

        if verbose_mode: print(ret_report)
    
    return rst


def main():
    dimension = 300
    shuffle = False
    training_ratio = 0.8
    metrics_path = "output/metrics_word2vec.csv"
    rst = classify_word2vec(dimension, shuffle, training_ratio)
    print_metrics(rst, metrics_path)


if __name__ == "__main__":
    main()
