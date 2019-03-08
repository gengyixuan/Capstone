
def gen_word2vec(dimension):
    raw_text_file = 'raw_text.txt'
    stop_words_file = 'stopwords.txt'
    raw_word_vec_file = 'glove.6B.' + str(dimension) + 'd.txt'
    word_vec_file = 'wordvec_d' + str(dimension) + '.txt'

    with open(raw_word_vec_file) as f_rawvec:
        raw_vec_data = dict()
        for line in f_rawvec:
            split = line.find(' ')
            raw_vec_data[line[:split]] = line[split + 1:]
        with open(stop_words_file, 'r') as f_stop:
            stop_words = [word.split()[0] for word in f_stop]
            with open(raw_text_file, 'r') as f_txt:
                vec_data = dict()
                for line in f_txt:
                    parts = line.split("|||")
                    sentence = parts[1].strip()
                    words = sentence.split(' ')
                    for word in words:
                        if word in raw_vec_data and word not in stop_words:
                            vec_data[word] = raw_vec_data[word]
                with open(word_vec_file, 'w+') as f_vec:
                    for word in vec_data:
                        f_vec.write("%s %s" % (word, vec_data[word]))
    print('Generated %d-dimension word2vec' % dimension)


def main():
    dimension = [50, 100, 200, 300]
    for d in dimension:
        gen_word2vec(d)
    print('Done')


if __name__ == '__main__':
    main()