# read in raw label,sentence text file
# for each sentence, make:
#       1. sentence level meta data (sentence char length, number of words)  -->  sentence_meta.txt
#       2. word level meta data (word POS tag, word length) -->  aligned_meta.txt

# sentence_meta.txt:
#   first line: whether each feature is numeric or not
#   all features per sentence in one line

# word_meta.txt:
#   first line: number of lines per input sentence features, whether each feature is numeric or not
#   X lines of word_meta features per input sentence


import nltk
from nltk import word_tokenize
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding('utf8')
# import ssl

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# nltk.download()

# nltk get tokens of the input sentence and get POS tag of each token


def data_model_meta(train_p, verbose_mode):
    def get_pos_tags(sentence):
        tokens = word_tokenize(sentence)
        pos_tag_tups = nltk.pos_tag(tokens)
        pos_tags = [tup[1] for tup in pos_tag_tups]
        return pos_tags, tokens

    def meta_preprocess(raw_text_file, verbose_mode):
        # raw_text_file = "topicclass/topicclass_train_small.txt"
        output_file_sen = "pre_processed_data/sentence_meta.txt"
        output_file_aligned = "pre_processed_data/aligned_meta.txt"

        with open(output_file_sen, "w+") as f_sen:
            f_sen.write("True True\n")
            with open(output_file_aligned, "w+") as f_ali:
                f_ali.write("False True\n")
                with open(raw_text_file) as f:
                    for line in f:
                        parts = line.split("|||")
                        sentence = parts[1].strip()
                        # get POS tags for each word
                        pos_tags, tokens = get_pos_tags(sentence)

                        # output sentence_meta
                        META_SENLEN = len(sentence)
                        META_NUMWORDS = len(tokens)
                        f_sen.write(str(META_SENLEN) + "\t" + str(META_NUMWORDS) + "\n")

                        # output aligned meta
                        for tag in pos_tags:
                            f_ali.write(tag + "\t")
                        f_ali.write("\n")
                        for token in tokens:
                            f_ali.write(str(len(token)) + "\t")
                        f_ali.write("\n")


    # preprocess data
    meta_preprocess(train_p, verbose_mode)

    #------------------------------------------------------------
    #------------------------------------------------------------

    MAX_SEN_LEN = 50
    # test_ratio = 1.0 - train_ratio

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
    with open(train_p) as f:
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


    X = get_one_hot(X_features, X_features_numeric, len(y))
    X = np.array(X)
    y = np.array(y)
    return X, y


if __name__ == "__main__":
    raw_text_file = "topicclass/topicclass_train_small.txt"
    X, Y = data_model_meta(raw_text_file, False)
    print(X.shape)
    print(Y.shape)
    # print(X)
    # print(Y)
