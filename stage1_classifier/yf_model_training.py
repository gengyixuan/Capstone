import sys
import numpy as np
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


postag_info_file = "word_embeddings/pos_tag_key.txt"
stop_words_file_path = "word_embeddings/stopwords.txt"
word_embedding_file_path = "word_embeddings/vector_subset.txt"
training_set_path = "pre_processed_data/pre_train.txt"
meta_path = "pre_processed_data/aligned_meta.txt"

vector_dim = 0
postag_dim = 0
sample_dim = 0

max_iter = 50


def get_postag_mapping():
    index = 0
    postag_index_dict = {}
    global postag_dim

    with open(postag_info_file, 'r') as infile:
        for oneline in infile.readlines():
            one_postag = oneline.strip('\n').split('|')[0]
            one_postag = one_postag[0: len(one_postag)-1]
            
            postag_index_dict[one_postag] = index
            index += 1
    
    postag_dim = len(postag_index_dict)
    return postag_index_dict


def load_stop_words():
    stop_set = set()
    with open(stop_words_file_path, 'r') as infile:
        for oneline in infile.readlines():
            one_word = oneline.strip('\n')
            stop_set.add(one_word)
    return stop_set


def get_word_embedding_dict():
    word_vector_dict = {}
    global vector_dim

    with open(word_embedding_file_path, 'r') as infile:
        for oneline in infile.readlines():
            one_list = oneline.strip().split()

            if vector_dim != len(one_list)-1:
                vector_dim = len(one_list) - 1

            one_word = one_list[0]
            one_vector = np.array(list(map(float, one_list[1:])))
            word_vector_dict[one_word] = one_vector

    return word_vector_dict


def get_one_training_example(one_sample, one_meta_sample, postag_index_dict, stop_set, word_vector_dict):
    one_sample_list = one_sample.strip('\n').split(' ||| ')
    one_label = int(one_sample_list[0])
    token_list = one_sample_list[1].split()

    token_vec_list = []
    global sample_dim

    for token_index in range(len(token_list)):
        one_token = token_list[token_index].lower()
        one_postag = one_meta_sample[0][token_index]
        one_token_len = int(one_meta_sample[1][token_index])

        #---------------------------------------------------------------------
        if (one_token in word_vector_dict) and (one_token not in stop_set):
            word_embed_vec_list = word_vector_dict[one_token]
            # print(word_embed_vec.shape)
            postag_vec = np.zeros(postag_dim)
            if one_postag in postag_index_dict:
                one_postag_index = postag_index_dict[one_postag]
                postag_vec[one_postag_index] = 1
            postag_vec_list = postag_vec

            #------------------------------------------------------------------
            one_token_vec = np.append(word_embed_vec_list, postag_vec_list)
            one_token_vec = np.append(one_token_vec, one_token_len)
            # one_token_vec = word_embed_vec_list

            if sample_dim == 0:
                sample_dim = one_token_vec.shape[0]

            token_vec_list.append(one_token_vec)
    
    #---------------------------------------------------------------------
    sample_vec = np.zeros(sample_dim)

    if len(token_vec_list) > 0:
        token_weight = 1.0 / len(token_vec_list)  # use average weight

        for index in range(len(token_vec_list)):
            sample_vec += token_vec_list[index] * token_weight
    
    # print(sample_vec.shape)
    return one_label, sample_vec


def get_training_data(postag_index_dict, stop_set, word_vector_dict):
    all_raw_data = open(training_set_path, 'r').readlines()
    all_meta_data = open(meta_path, 'r').readlines()

    meta_data = []
    for index in range(1, len(all_meta_data), 2):
        postag_line = all_meta_data[index].split('\t')
        token_len_line = all_meta_data[index+1].split('\t')
        meta_data.append([postag_line, token_len_line])

    X_list = []
    Y_list = []

    for index in range(len(all_raw_data)):
        sample_label, sample_vec = get_one_training_example(all_raw_data[index], 
            meta_data[index], postag_index_dict, stop_set, word_vector_dict)
        
        X_list.append(sample_vec)
        Y_list.append(sample_label)
    
    X_arr = np.array(X_list)
    Y_arr = np.array(Y_list)

    # print(X_arr.shape)
    # print(Y_arr.shape)
    return X_arr, Y_arr


def model_training(X_arr, Y_arr, test_ratio, verbose_mode):
    X_train, X_test, y_train, y_test = train_test_split(X_arr, Y_arr, test_size=test_ratio)

    clf = MLPClassifier(
        hidden_layer_sizes=(1000, 1000),
        activation='relu',
        solver='adam',
        alpha=0.0002,
        batch_size='auto',
        learning_rate='constant',
        learning_rate_init=0.01,
        power_t=0.5,
        max_iter=max_iter,
        shuffle=True,
        random_state=None,
        tol=0.0001,
        verbose=verbose_mode,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9, beta_2=0.999, epsilon=1e-08,
        n_iter_no_change=10)

    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)

    res = sklearn.metrics.classification_report(y_test, y_predict)
    # print(res)
    return res


def classify_concat(training_ratio, verbose_mode):
    postag_index_dict = get_postag_mapping()
    stop_set = load_stop_words()
    word_vector_dict = get_word_embedding_dict()

    X_arr, Y_arr = get_training_data(postag_index_dict, stop_set, word_vector_dict)
    res = model_training(X_arr, Y_arr, 1.0-training_ratio, verbose_mode)
    return res


if __name__ == "__main__":
    _ = classify_concat(0.8)
