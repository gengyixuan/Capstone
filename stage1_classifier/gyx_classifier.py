# read three files: raw_text_file, aligned_meta, sentence_meta
# for each (label,sentence,sentence_meta,sen_meta_numeric,aligned_meta[],aligned_meta_numeric):
#       pad sentence length / cap at max length
#       for non-numeric features, convert to one-hot
#       append them into one-huge training vector
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np


def classify_meta(raw_text_file, train_ratio, verbose_mode):
    MAX_SEN_LEN = 50
    test_ratio = 1.0 - train_ratio

    X_features = []
    X_features_numeric = []
    y = []

    # raw_text_file = "topicclass/topicclass_train_small.txt"
    output_file_sen = "pre_processed_data/sentence_meta.txt"
    output_file_aligned = "pre_processed_data/aligned_meta.txt"

    sen_meta_numeric = []
    aligned_meta_numeric = []

    if verbose_mode: print("start preprocessing")

    # read sentence_meta into feature vector
    with open(output_file_sen, "r") as f_sen:
        num_features = 0
        first_line = True
        for line in f_sen:
            # if first line, get sen_meta_numeric
            if first_line:
                first_line = False
                line = line[:-1] # get rid of \n
                bools = line.split()
                num_features = len(bools)
                X_features = [[] for _ in range(num_features)]
                sen_meta_numeric = [True if b == 'True' else False for b in bools]
                X_features_numeric = sen_meta_numeric
            else:
                line = line[:-1]
                features = line.split("\t")
                for i in range(num_features):
                    X_features[i].append(float(features[i]) if sen_meta_numeric[i] else features[i])

    # read word_meta into feature vector
    with open(output_file_aligned, "r") as f_ali:
        num_features = 0
        num_sen_meta_features = len(X_features)
        line_id = -1
        for line in f_ali:
            # if first line, get sen_meta_numeric
            if line_id == -1:
                line = line[:-1] # get rid of \n
                bools = line.split()
                num_features = len(bools)
                X_features += [[] for _ in range(num_features * MAX_SEN_LEN)]
                aligned_meta_numeric = [True if b == 'True' else False for b in bools]

                for i in range(num_features):
                    X_features_numeric += [aligned_meta_numeric[i] for _ in range(MAX_SEN_LEN)]
            else:
                # line_id starts at 0
                line = line[:-1]
                isNumeric = aligned_meta_numeric[line_id % num_features]
                features = line.split("\t")
                for i in range(MAX_SEN_LEN):
                    feature_id = num_sen_meta_features + line_id % num_features * MAX_SEN_LEN + i

                    if i < len(features) - 1:
                        X_features[feature_id].append(float(features[i]) if isNumeric else features[i])
                    else:
                        X_features[feature_id].append(0.0 if isNumeric else "<PAD>")

            line_id += 1

    # read labels
    with open(raw_text_file) as f:
        for line in f:
            parts = line.split("|||")
            label = parts[0].strip()
            y.append(label)



    def get_one_hot(X_features, numeric, num_data):
        enc = OneHotEncoder(handle_unknown='ignore')
        X = [[] for _ in range(num_data)]
        for i,feature in enumerate(X_features):
            if numeric[i]:
                for x in range(num_data):
                    X[x].append(feature[x])
            else:
                # get one hot encoding
                one_d_feature = [[f] for f in feature]
                enc.fit(one_d_feature)
                one_hot_feature = enc.transform(one_d_feature).toarray()
                for x in range(num_data):
                    X[x] = X[x] + one_hot_feature[x].tolist() # concat one-hot feature to feature vec, for each data point
        return X


    # preprocess data
    X = get_one_hot(X_features, X_features_numeric, len(y))
    X = np.array(X)
    y = np.array(y)

    # X_train = X[:100,:]
    # y_train = y[:100]
    # X_test = X[100:,:]
    # y_test = y[100:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio)
    # train model
    from sklearn.naive_bayes import MultinomialNB

    clf = MultinomialNB().fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # visualize evaluation
    from sklearn.metrics import confusion_matrix
    conf_mat = confusion_matrix(y_test, y_pred)
    if verbose_mode: print(conf_mat)
    from sklearn import metrics
    if verbose_mode: print(metrics.classification_report(y_test, y_pred))
    ret_report = metrics.classification_report(y_test, y_pred)
    return ret_report


if __name__ == "__main__":
    raw_text_path = "topicclass/topicclass_train_small.txt"
    _ = classify_meta(raw_text_path)