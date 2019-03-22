import sys
import io
import numpy as np


vector_dim = 0
postag_dim = 0
sample_dim = 0


def data_model_combine(train_p, verbose_mode):
    word_dict = {}

    # input from embedding files
    word_embedding_file = "word_embeddings/glove.6B.300d.txt"

    # input intermediate files
    postag_info_file = "word_embeddings/pos_tag_key.txt"
    stop_words_file_path = "word_embeddings/stopwords.txt"

    # intermediate files input from GYX
    training_set_path = "pre_processed_data/pre_train.txt"
    meta_path = "pre_processed_data/aligned_meta.txt"

    # output intermediate files
    word_subset_file = "word_embeddings/vector_subset.txt"
    label_path = "word_embeddings/label_mapping.txt"
    output_path = "pre_processed_data/pre_train.txt"
    

    def generate_mapping(train_path, verbose_mode):
        label_list = []

        with open(train_path, 'r', encoding='utf-8') as infile:
            for oneline in infile.readlines():
                onelabel = oneline.strip().split('|||')[0].strip()

                if onelabel not in label_list:
                    label_list.append(onelabel)

        label_dict = {}
        with open(label_path, 'w') as outfile1:
            for index in range(len(label_list)):
                if verbose_mode: print(str(label_list[index]) + ": " + str(index))
                
                label_dict[label_list[index]] = index

                oneline = str(label_list[index]) + " ||| " + str(index)
                outfile1.writelines(oneline + '\n')

        # for index in range(len(pre_process_input_file_list)):
        input_file_path = train_path
        output_file_path = output_path

        all_data = open(input_file_path, 'r').readlines()

        with open(output_file_path, 'w') as outfile:
            for line_index in range(len(all_data)):
                oneline = all_data[line_index]

                if line_index % 5000 == 0:
                    if verbose_mode: print(line_index)

                one_list = oneline.strip().split('|||')
                text_label = one_list[0].strip()

                if text_label not in label_dict.keys():
                    text_label = "Media and drama"

                try:
                    digit_label = label_dict[text_label]
                    out_one_line = str(digit_label) + " |||" + one_list[1]
                    outfile.writelines(out_one_line + '\n')
                except:
                    if verbose_mode: print("err at line: " + str(line_index))
                    if verbose_mode: print("err key: " + str(text_label))


    def extract_word_cnt(file_name):
        # global word_dict

        with open(file_name, 'r', encoding='utf-8') as infile:
            for oneline in infile.readlines():
                one_sentence = oneline.strip().split('|||')[1].strip()

                for one_word in one_sentence.split():
                    word = one_word.strip().lower()
                    word_dict[word] = 1


    def save_subset_vector(word_dict, verbose_mode):
        fin = io.open(word_embedding_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
        # n, d = map(int, fin.readline().split())
        data = {}
        hit_dict = {}

        with open(word_subset_file, 'w', encoding='utf-8') as outfile:
            for line in fin:
                tokens = line.rstrip().split(' ')
                one_word = tokens[0].lower()

                if (one_word in word_dict) and (one_word not in hit_dict):
                    token_vec = list(map(float, tokens[1:]))
                    token_vec_str = ""

                    for element_index in range(len(token_vec)):
                        token_vec_str += " " + str(token_vec[element_index])

                    outfile.writelines(str(one_word) + token_vec_str + "\n")
                    hit_dict[one_word] = 1
        
        if verbose_mode: print("hit: " + str(len(hit_dict)))
        return hit_dict


    def extract_vector_subset(verbose_mode):
        extract_word_cnt(output_path)

        if verbose_mode: print("\nTotal unique number of words all sets: " + str(len(word_dict)))

        hit_dict = save_subset_vector(word_dict, verbose_mode)

        miss_list = []
        for one_key in word_dict:
            if one_key not in hit_dict:
                miss_list.append(one_key)
        
        for index in range(len(miss_list)):
            if verbose_mode: print(miss_list[index].encode('utf-8'))
            if index > 10:
                break


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

        with open(word_subset_file, 'r') as infile:
            for oneline in infile.readlines():
                one_list = oneline.strip().split()

                if vector_dim != len(one_list)-1:
                    vector_dim = len(one_list) - 1

                one_word = one_list[0]
                one_vector = np.array(list(map(float, one_list[1:])))
                word_vector_dict[one_word] = one_vector

        return word_vector_dict


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

        return X_arr, Y_arr


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


    def concat_preprocess(train_p, verbose_mode):
        generate_mapping(train_p, verbose_mode)
        extract_vector_subset(verbose_mode)

    concat_preprocess(train_p, verbose_mode)
    postag_index_dict = get_postag_mapping()
    stop_set = load_stop_words()
    word_vector_dict = get_word_embedding_dict()
    X, Y = get_training_data(postag_index_dict, stop_set, word_vector_dict)
    return X, Y


if __name__ == "__main__":
    train_p = "topicclass/topicclass_train_small.txt"
    X, Y = data_model_combine(train_p, True)
    print(X.shape)
    print(Y.shape)
    # print(X)
    # print(Y)
