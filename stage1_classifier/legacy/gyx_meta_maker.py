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
# import ssl

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# nltk.download()

# nltk get tokens of the input sentence and get POS tag of each token
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


if __name__ == "__main__":
    raw_text_file = "topicclass/topicclass_train_small.txt"
    meta_preprocess(raw_text_file)
