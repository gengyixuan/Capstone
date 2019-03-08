import numpy as np
from sklearn.pipeline import Pipeline
from sklearn import *
from collections import defaultdict


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


if __name__ == "__main__":
    DIMENSION = 200
    SHUFFLE = False
    WORD2VEC_PATH = "wordvec_d%d.txt" % DIMENSION
    RAW_TEXT_PATH = "raw_text.txt"
    METRICS_PATH = "metrics_word2vec.csv"

    # Read pre-trained list
    with open(WORD2VEC_PATH, "r") as lines:
        w2v = {line.split()[0]: np.array(list(map(float, line.split()[1:])))
               for line in lines}

    # Read from raw text
    label = []
    data = []
    with open(RAW_TEXT_PATH, "r") as lines:
        for line in lines:
            segs = line.split("|||")
            label.append(segs[0])
            data.append(segs[1].split())
    data = np.array(data)
    label = np.array(label)

    # Build pipeline
    models = dict()

    models['SVM'] = Pipeline([
        ('word2vec vectorizer', MeanEmbeddingVectorizer(w2v, DIMENSION)),
        ('svm', svm.LinearSVC(random_state=0, tol=1e-5))
    ])
    models['Neural Network'] = Pipeline([
        ('word2vec vectorizer', MeanEmbeddingVectorizer(w2v, DIMENSION)),
        ('neural network', neural_network.MLPClassifier(verbose=True))
    ])
    models['Extra Tree'] = Pipeline([
        ('word2vec vectorizer', MeanEmbeddingVectorizer(w2v, DIMENSION)),
        ('extra tree', ensemble.ExtraTreesClassifier(n_estimators=200))
    ])

    # Shuffle dataset
    if SHUFFLE:
        data_label = np.c_[data.reshape(len(data), -1), label.reshape(len(label), -1)]
        np.random.shuffle(data_label)
        data = data_label[:, :data.size//len(data)].reshape(data.shape)
        label = data_label[:, data.size//len(data):].reshape(label.shape)

    # Split dataset
    train_data = data[:500]
    train_label = label[:500]
    test_data = data[500:]
    test_label = label[500:]

    # Train and test
    rst = dict()
    for model in models:
        rst[model] = dict()
        models[model].fit(train_data, train_label)
        test_pred = models[model].predict(test_data)
        entry = metrics.precision_recall_fscore_support(test_label, test_pred)
        accuracy, recall, f1, support = entry
        accuracy = np.mean(accuracy)
        recall = np.mean(recall)
        f1 = np.mean(f1)
        rst[model] = accuracy, recall, f1

    # Store metrics
    with open(METRICS_PATH, "w+") as f_metrics:
        f_metrics.write("Model, Accuracy, Recall, F1\n")
        for model in models:
            f_metrics.write("%s, %.4f, %.4f, %.4f\n" %
                            (model,
                             rst[model][0],
                             rst[model][1],
                             rst[model][2]))
