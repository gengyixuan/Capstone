
dimension = 50

raw_text_file = 'raw_text.txt'
raw_word_vec_file = 'glove.6B.' + str(dimension) + 'd.txt'
word_vec_file = 'wordvec_d' + str(dimension) + '.txt'

with open(raw_text_file, 'r') as f_txt:
    with open(raw_word_vec_file) as f_rawvec:
        raw_vec_data = dict()
        for line in f_rawvec:
            split = line.find(' ')
            raw_vec_data[line[:split]] = line[split + 1:]
        vec_data = dict()
        for line in f_txt:
            parts = line.split("|||")
            sentence = parts[1].strip()
            words = sentence.split(' ')
            for word in words:
                if word in raw_vec_data:
                    vec_data[word] = raw_vec_data[word]
        with open(word_vec_file, 'w+') as f_vec:
            for word in vec_data:
                f_vec.write("%s\t%s" % (word, vec_data[word]))
